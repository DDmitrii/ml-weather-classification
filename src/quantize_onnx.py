# src/quantize_onnx.py
import numpy as np
import torch
import typing
import omegaconf
import onnx
from pathlib import Path
from hydra import initialize, compose
from torch.utils.data import DataLoader

from src.data import WeatherDataset, get_val_transforms
from src.model import WeatherClassifier

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

FP32_PATH   = "exports/convnext_tiny_fp32.onnx"
INT8_PATH   = "exports/convnext_tiny_int8.onnx"
IMG_SIZE    = 224
CALIB_SIZE  = 128   # количество картинок для калибровки


def get_calibration_data(cfg) -> list[np.ndarray]:
    """Берём CALIB_SIZE картинок из val для калибровки."""
    val_ds = WeatherDataset(
        root=cfg.data.val_dir,
        class_names=list(cfg.data.class_names),
        transform=get_val_transforms(IMG_SIZE),
    )
    loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    batches = []
    total = 0
    for batch in loader:
        x = batch[0].numpy()
        batches.append(x)
        total += x.shape[0]
        if total >= CALIB_SIZE:
            break

    return batches


if __name__ == "__main__":
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train", overrides=["~experiments"])

    Path("exports").mkdir(exist_ok=True)

    # ── Шаг 1: pre-process (shape inference) ──────────────────────────
    FP32_PREP_PATH = "exports/convnext_tiny_fp32_prep.onnx"
    print("🔧 Запускаю shape inference...")
    quant_pre_process(FP32_PATH, FP32_PREP_PATH, skip_optimization=False)
    print("✅ Shape inference готов")

    # ── Шаг 2: CalibrationDataReader ──────────────────────────────────
    print(f"📂 Собираю калибровочные данные ({CALIB_SIZE} картинок)...")
    calib_batches = get_calibration_data(cfg)

    class WeatherCalibReader(CalibrationDataReader):
        def __init__(self, batches):
            self.batches = iter(batches)

        def get_next(self):
            try:
                batch = next(self.batches)
                return {"image": batch}
            except StopIteration:
                return None

    # ── Шаг 3: Static quantization ────────────────────────────────────
    print("⚙️  Квантизую модель (static int8)...")
    quantize_static(
        model_input=FP32_PREP_PATH,
        model_output=INT8_PATH,
        calibration_data_reader=WeatherCalibReader(calib_batches),
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True,
        nodes_to_exclude=[
            n.name for n in onnx.load(FP32_PREP_PATH).graph.node
            if n.op_type == "LayerNormalization"
        ],
    )
    print(f"✅ Квантизованная модель сохранена: {INT8_PATH}")

    # ── Шаг 4: Сравниваем размеры ─────────────────────────────────────
    fp32_mb = Path(FP32_PATH).stat().st_size / 1024 / 1024
    int8_mb = Path(INT8_PATH).stat().st_size / 1024 / 1024
    print(f"\n📦 Размер fp32 : {fp32_mb:.1f} MB")
    print(f"📦 Размер int8 : {int8_mb:.1f} MB")
    print(f"📉 Сжатие      : {fp32_mb / int8_mb:.1f}x")

    # ── Шаг 5: Benchmark latency ──────────────────────────────────────
    import onnxruntime as ort
    import time

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    sess_fp32 = ort.InferenceSession(FP32_PATH,  providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8_PATH,  providers=["CPUExecutionProvider"])

    def benchmark(sess, n=50):
        # прогрев
        for _ in range(5):
            sess.run(["logits"], {"image": dummy})
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(["logits"], {"image": dummy})
        return (time.perf_counter() - t0) / n * 1000  # ms

    print("\n⏱️  Замеряю latency (50 итераций, batch=1, CPU)...")
    lat_fp32 = benchmark(sess_fp32)
    lat_int8 = benchmark(sess_int8)
    print(f"   fp32 latency : {lat_fp32:.1f} ms")
    print(f"   int8 latency : {lat_int8:.1f} ms")
    print(f"   Ускорение    : {lat_fp32 / lat_int8:.1f}x")

    # ── Шаг 6: Accuracy на val ────────────────────────────────────────
    print("\n📊 Считаю accuracy на val...")
    with initialize(config_path="../configs", version_base=None):
        cfg2 = compose(config_name="train", overrides=["~experiments"])

    val_ds = WeatherDataset(
        root=cfg2.data.val_dir,
        class_names=list(cfg2.data.class_names),
        transform=get_val_transforms(IMG_SIZE),
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    def eval_onnx(sess):
        correct, total = 0, 0
        for batch in val_loader:
            x = batch[0].numpy()
            y = batch[1].numpy()
            logits = sess.run(["logits"], {"image": x})[0]
            preds  = logits.argmax(axis=1)
            correct += (preds == y).sum()
            total   += len(y)
        return correct / total

    acc_fp32 = eval_onnx(sess_fp32)
    acc_int8 = eval_onnx(sess_int8)
    print(f"   fp32 accuracy : {acc_fp32:.4f}")
    print(f"   int8 accuracy : {acc_int8:.4f}")
    print(f"   Потеря        : {(acc_fp32 - acc_int8):.4f}")