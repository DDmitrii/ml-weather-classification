# src/quantize_onnx.py
import argparse
import numpy as np
import torch
import typing
import omegaconf
from pathlib import Path
from hydra import initialize, compose
from torch.utils.data import DataLoader
from onnxruntime.quantization import quantize_dynamic

from src.data import WeatherDataset, get_val_transforms

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

IMG_SIZE   = 224
CALIB_SIZE = 128

CONFIGS = {
    "teacher": {
        "fp32": "exports/convnext_tiny_fp32.onnx",
        "int8": "exports/convnext_tiny_int8.onnx",
    },
    "student": {
        "fp32": "exports/mobilenet_v3_fp32.onnx",
        "int8": "exports/mobilenet_v3_int8.onnx",
    },
}


def get_calibration_data(cfg) -> list[np.ndarray]:
    val_ds = WeatherDataset(
        root=cfg.data.val_dir,
        class_names=list(cfg.data.class_names),
        transform=get_val_transforms(IMG_SIZE),
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    batches, total = [], 0
    for batch in loader:
        x = batch[0].numpy()
        batches.append(x)
        total += x.shape[0]
        if total >= CALIB_SIZE:
            break
    return batches


if __name__ == "__main__":
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader,
        QuantFormat, QuantType,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="teacher",
        help="Какую модель квантизовать",
    )
    args, _ = parser.parse_known_args()

    c = CONFIGS[args.model]
    FP32_PATH      = c["fp32"]
    INT8_PATH      = c["int8"]
    FP32_PREP_PATH = FP32_PATH.replace(".onnx", "_prep.onnx")

    print(f"🚀 Квантизуем: {args.model} → {INT8_PATH}\n")

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["~experiments"])

    Path("exports").mkdir(exist_ok=True)

    # ── Шаг 2: Калибровочные данные ───────────────────────────────
    print(f"📂 Собираю калибровочные данные ({CALIB_SIZE} картинок)...")
    calib_batches = get_calibration_data(cfg)

    class WeatherCalibReader(CalibrationDataReader):
        def __init__(self, batches):
            self.batches = iter(batches)
        def get_next(self):
            try:
                return {"image": next(self.batches)}
            except StopIteration:
                return None

    # ── Шаг 3: Static quantization ────────────────────────────────
    print("⚙️  Квантизую модель (dynamic int8)...")
    quantize_dynamic(
        model_input=FP32_PATH,  # ← напрямую fp32, без _prep
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,
    )
    print(f"✅ Квантизованная модель сохранена: {INT8_PATH}")

    # ── Шаг 4: Размеры ────────────────────────────────────────────
    fp32_mb = Path(FP32_PATH).stat().st_size / 1024 / 1024
    int8_mb = Path(INT8_PATH).stat().st_size / 1024 / 1024
    print(f"\n📦 Размер fp32 : {fp32_mb:.1f} MB")
    print(f"📦 Размер int8 : {int8_mb:.1f} MB")
    print(f"📉 Сжатие      : {fp32_mb / int8_mb:.1f}x")

    # ── Шаг 5: Latency benchmark ──────────────────────────────────
    import onnxruntime as ort
    import time

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    sess_fp32 = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])

    def benchmark(sess, n=50):
        for _ in range(5):
            sess.run(["logits"], {"image": dummy})
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(["logits"], {"image": dummy})
        return (time.perf_counter() - t0) / n * 1000

    print("\n⏱️  Замеряю latency (50 итераций, batch=1, CPU)...")
    lat_fp32 = benchmark(sess_fp32)
    lat_int8 = benchmark(sess_int8)
    print(f"   fp32 latency : {lat_fp32:.1f} ms")
    print(f"   int8 latency : {lat_int8:.1f} ms")
    print(f"   Ускорение    : {lat_fp32 / lat_int8:.1f}x")

    # ── Шаг 6: Accuracy на val ────────────────────────────────────
    print("\n📊 Считаю accuracy на val...")
    val_ds = WeatherDataset(
        root=cfg.data.val_dir,
        class_names=list(cfg.data.class_names),
        transform=get_val_transforms(IMG_SIZE),
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    def eval_onnx(sess):
        correct, total = 0, 0
        for batch in val_loader:
            x = batch[0].numpy()
            y = batch[1].numpy()
            preds = sess.run(["logits"], {"image": x})[0].argmax(axis=1)
            correct += (preds == y).sum()
            total   += len(y)
        return correct / total

    acc_fp32 = eval_onnx(sess_fp32)
    acc_int8 = eval_onnx(sess_int8)
    print(f"   fp32 accuracy : {acc_fp32:.4f}")
    print(f"   int8 accuracy : {acc_int8:.4f}")
    print(f"   Потеря        : {acc_fp32 - acc_int8:.4f}")