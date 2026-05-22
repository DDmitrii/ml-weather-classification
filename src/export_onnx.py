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
from src.model import WeatherClassifier

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
        "ckpt":        "checkpoints/fold2/epoch=21-val_f1=0.9800.ckpt",
        "onnx":        "exports/convnext_tiny_fp32.onnx",
        "model_name":  "convnext_tiny",
        "is_lightning": True,
    },
    "student": {
        "ckpt":        "exports/mobilenet_v3_student.pt",
        "onnx":        "exports/mobilenet_v3_fp32.onnx",
        "model_name":  "mobilenetv3_small_100",
        "is_lightning": False,
    },
}


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_model(cfg, model_type: str) -> nn.Module:
    c = CONFIGS[model_type]
    num_classes = len(cfg.data.class_names)

    if c["is_lightning"]:
        # Teacher — PyTorch Lightning checkpoint
        train_ds = WeatherDataset(
            root=cfg.data.train_dir,
            class_names=list(cfg.data.class_names),
            transform=get_train_transforms(IMG_SIZE),
        )
        class_weights = train_ds.get_class_weights()
        pl_model = WeatherClassifier.load_from_checkpoint(
            c["ckpt"],
            cfg=cfg,
            class_weights=class_weights,
            weights_only=False,
        )
        pl_model.eval()
        model = pl_model.model  # timm backbone
        print(f"🔍 Тип model.model: {type(model)}")
        print(f"🔍 Дочерние модули: {[name for name, _ in model.named_children()]}")
    else:
        # Student — обычный torch checkpoint
        model = timm.create_model(
            c["model_name"],
            pretrained=False,
            num_classes=num_classes,
        )
        ckpt = torch.load(c["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"🔍 Тип модели: {type(model)}")
        print(f"🔍 Дочерние модули: {[name for name, _ in model.named_children()]}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="teacher",
        help="Какую модель экспортировать",
    )
    args, _ = parser.parse_known_args()

    c = CONFIGS[args.model]
    ONNX_PATH = c["onnx"]

    print(f"🚀 Экспортируем: {args.model} → {ONNX_PATH}\n")

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["~experiments"])

    model = load_model(cfg, args.model)
    model.to("cpu")

    wrapper = ExportWrapper(model)
    wrapper.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    Path("exports").mkdir(exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy,
        ONNX_PATH,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"✅ Экспортировано: {ONNX_PATH}")

    # Валидация графа
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX граф валиден")

    # Проверка инференса
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    out  = sess.run(["logits"], {"image": dummy.numpy()})
    print(f"✅ OnnxRuntime inference OK — logits shape: {out[0].shape}")

    # Размер файла
    size_mb = Path(ONNX_PATH).stat().st_size / 1024 / 1024
    print(f"📦 Размер fp32 модели: {size_mb:.1f} MB")

    # Расхождение PyTorch vs ONNX
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()
    max_diff = np.abs(pt_out - out[0]).max()
    print(f"🔎 Максимальное расхождение PyTorch vs ONNX: {max_diff:.6f}")