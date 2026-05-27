import argparse
import numpy as np
import torch
import typing
import omegaconf
from pathlib import Path
from hydra import initialize, compose
from torch.utils.data import DataLoader
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.data import WeatherDataset, get_val_transforms

torch.serialization.add_safe_globals([
    omegaconf.dictconfig.DictConfig,
    omegaconf.listconfig.ListConfig,
    omegaconf.base.ContainerMetadata,
    typing.Any,
])

IMG_SIZE = 224

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["teacher", "student"],
        default="student",
        help="Какую модель квантизовать",
    )
    args, _ = parser.parse_known_args()

    c = CONFIGS[args.model]
    FP32_PATH = c["fp32"]
    INT8_PATH = c["int8"]

    print(f"Квантизуем: {args.model} → {INT8_PATH}\n")

    with initialize(config_path="../configs", version_base=None):
        cfg = compose(
            config_name="train",
            overrides=["experiments=focal_loss_multihead"],
        )

    Path("exports").mkdir(exist_ok=True)

    print("⚙️  Квантизую модель (dynamic int8)...")
    quantize_dynamic(
        model_input=FP32_PATH,
        model_output=INT8_PATH,
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,
    )
    print(f"Квантизованная модель сохранена: {INT8_PATH}")

    fp32_mb = Path(FP32_PATH).stat().st_size / 1024 / 1024
    int8_mb = Path(INT8_PATH).stat().st_size / 1024 / 1024
    print(f"\nРазмер fp32 : {fp32_mb:.1f} MB")
    print(f"Размер int8 : {int8_mb:.1f} MB")
    print(f"Сжатие      : {fp32_mb / int8_mb:.1f}x")

    import onnxruntime as ort
    import time

    is_teacher = (args.model == "teacher")
    output_names = ["logits_dn", "logits_wt"] if is_teacher else ["logits"]

    dummy = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    sess_fp32 = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])

    def benchmark(sess, n=50):
        for _ in range(5):
            sess.run(output_names, {"image": dummy})
        t0 = time.perf_counter()
        for _ in range(n):
            sess.run(output_names, {"image": dummy})
        return (time.perf_counter() - t0) / n * 1000

    print("\nЗамеряю latency (50 итераций, batch=1, CPU)...")
    lat_fp32 = benchmark(sess_fp32)
    lat_int8 = benchmark(sess_int8)
    print(f"   fp32 latency : {lat_fp32:.1f} ms")
    print(f"   int8 latency : {lat_int8:.1f} ms")
    print(f"   Ускорение    : {lat_fp32 / lat_int8:.1f}x")

    print("\nСчитаю accuracy на val...")
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
            if is_teacher:
                logits_dn, logits_wt = sess.run(output_names, {"image": x})
                pred_dn = logits_dn.argmax(axis=1)
                pred_wt = logits_wt.argmax(axis=1)
                from src.data.dataset import COMBO_TO_FINAL
                preds = np.array([
                    COMBO_TO_FINAL.get((dn, wt), 0)
                    for dn, wt in zip(pred_dn, pred_wt)
                ])
            else:
                preds = sess.run(output_names, {"image": x})[0].argmax(axis=1)
            correct += (preds == y).sum()
            total += len(y)
        return correct / total

    acc_fp32 = eval_onnx(sess_fp32)
    acc_int8 = eval_onnx(sess_int8)
    print(f"   fp32 accuracy : {acc_fp32:.4f}")
    print(f"   int8 accuracy : {acc_int8:.4f}")
    print(f"   Потеря        : {acc_fp32 - acc_int8:.4f}")