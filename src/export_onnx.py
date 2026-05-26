# src/export_onnx.py
import argparse
import torch
import torch.nn as nn
import timm
import typing
import omegaconf
from pathlib import Path
from hydra import initialize, compose

from src.data import WeatherDataset, get_train_transforms
from src.model.train import WeatherClassifier, WeatherClassifierMultiHead

torch.set_float32_matmul_precision('high')
torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

IMG_SIZE = 224

CONFIGS = {
    "teacher": {
        "ckpt":       "checkpoints/epoch=17-val_f1=0.9921.ckpt",
        "onnx":       "exports/convnext_tiny_fp32.onnx",
        "multihead":  True,
    },
    "student": {
        "ckpt":       "exports/mobilenet_v3_student.pt",
        "onnx":       "exports/mobilenet_v3_fp32.onnx",
        "model_name": "mobilenetv3_small_100",
        "multihead":  False,
    },
}


class MultiHeadExportWrapper(nn.Module):
    """Оборачивает WeatherClassifierMultiHead: возвращает 2 выхода."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        logits_dn, logits_wt = self.model.backbone(x), None
        # Используем forward модели напрямую
        return self.model(x)  # -> (logits_dn, logits_wt)


class SingleHeadExportWrapper(nn.Module):
    """Оборачивает одноголовую модель: один выход logits."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model(cfg, model_type: str) -> tuple[nn.Module, bool]:
    c = CONFIGS[model_type]

    if model_type == "teacher":
        train_ds = WeatherDataset(
            root=cfg.data.train_dir,
            class_names=list(cfg.data.class_names),
            transform=get_train_transforms(IMG_SIZE),
        )
        class_weights = train_ds.get_class_weights()
        pl_model = WeatherClassifierMultiHead.load_from_checkpoint(
            c["ckpt"],
            cfg=cfg,
            class_weights=class_weights,
            weights_only=False,
        )
        pl_model.eval()
        return pl_model, True  # is_multihead=True

    else:
        num_classes = len(cfg.data.class_names)
        model = timm.create_model(
            c["model_name"],
            pretrained=False,
            num_classes=num_classes,
        )
        ckpt = torch.load(c["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model, False  # is_multihead=False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["teacher", "student"], default="teacher")
    args, _ = parser.parse_known_args()

    c = CONFIGS[args.model]
    ONNX_PATH = c["onnx"]
    print(f"🚀 Экспортируем: {args.model} → {ONNX_PATH}\n")

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiments=focal_loss_multihead"],
        )

    model, is_multihead = load_model(cfg, args.model)
    model.to("cpu").eval()

    if is_multihead:
        wrapper = MultiHeadExportWrapper(model)
        output_names = ["logits_dn", "logits_wt"]
        dynamic_axes = {
            "image":     {0: "batch_size"},
            "logits_dn": {0: "batch_size"},
            "logits_wt": {0: "batch_size"},
        }
    else:
        wrapper = SingleHeadExportWrapper(model)
        output_names = ["logits"]
        dynamic_axes = {
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        }

    wrapper.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    Path("exports").mkdir(exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        dynamo=False,
    )
    print(f"✅ Экспортировано: {ONNX_PATH}")

    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX граф валиден")

    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out  = sess.run(output_names, {"image": dummy.numpy()})

    for name, tensor in zip(output_names, out):
        print(f"✅ OnnxRuntime inference OK — {name} shape: {tensor.shape}")

    size_mb = Path(ONNX_PATH).stat().st_size / 1024 / 1024
    print(f"📦 Размер fp32 модели: {size_mb:.1f} MB")

    with torch.no_grad():
        pt_out = wrapper(dummy)
        if is_multihead:
            pt_out = [o.numpy() for o in pt_out]
        else:
            pt_out = [pt_out.numpy()]

    for name, pt, onnx_t in zip(output_names, pt_out, out):
        max_diff = np.abs(pt - onnx_t).max()
        print(f"🔎 Макс. расхождение PyTorch vs ONNX [{name}]: {max_diff:.6f}")