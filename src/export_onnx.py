# src/export_onnx.py
import torch
import torch.nn as nn
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

CHECKPOINT = "checkpoints/fold2/epoch=21-val_f1=0.9800.ckpt"
ONNX_PATH  = "exports/convnext_tiny_fp32.onnx"
IMG_SIZE   = 224


class ExportWrapper(nn.Module):
    def __init__(self, pl_model):
        super().__init__()
        self.model = pl_model.model  # timm backbone + classifier head

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["~experiments"])

    # Загружаем модель
    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(IMG_SIZE),
    )
    class_weights = train_ds.get_class_weights()

    pl_model = WeatherClassifier.load_from_checkpoint(
        CHECKPOINT,
        cfg=cfg,
        class_weights=class_weights,
        weights_only=False,
    )
    pl_model.eval()
    pl_model.to("cpu")

    # Диагностика структуры
    print(f"🔍 Тип model.model: {type(pl_model.model)}")
    print(f"🔍 Дочерние модули: {[name for name, _ in list(pl_model.model.named_children())]}")

    wrapper = ExportWrapper(pl_model)
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
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
        dynamo=False,  # ← принудительно старый экспортёр
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

    # Сравниваем с PyTorch выходом
    with torch.no_grad():
        pt_out = wrapper(dummy).numpy()
    max_diff = np.abs(pt_out - out[0]).max()
    print(f"🔎 Максимальное расхождение PyTorch vs ONNX: {max_diff:.6f}")