# src/api/predictor.py
import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.api.schemas import CLASS_NAMES
from src.data.dataset import COMBO_TO_FINAL

logger = logging.getLogger(__name__)

IMG_SIZE    = 224
MAX_CACHE   = 256        # максимум записей в LRU-кэше
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ONNX_PATHS = {
    "teacher": "exports/convnext_tiny_fp32.onnx",
    "student": "exports/mobilenet_v3_fp32.onnx",
}


# ── LRU-кэш ─────────────────────────────────────────────────────────────────

class LRUCache:
    """Простой thread-safe LRU-кэш на OrderedDict."""

    def __init__(self, capacity: int):
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._capacity = capacity

    def get(self, key: str) -> dict | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: dict) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def __len__(self) -> int:
        return len(self._cache)


# ── Препроцессинг ────────────────────────────────────────────────────────────

def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _image_hash(image: Image.Image) -> str:
    """MD5-хэш пикселей изображения для кэш-ключа."""
    return hashlib.md5(image.tobytes()).hexdigest()


# ── Predictor ────────────────────────────────────────────────────────────────

class WeatherPredictor:
    """
    Загружает ONNX-модели при старте, выполняет инференс с LRU-кэшем.
    Инференс вынесен в threadpool через run_in_executor (см. app.py).
    """

    def __init__(self):
        self._sessions: dict[str, ort.InferenceSession] = {}
        self._caches: dict[str, LRUCache] = {}

        for model_name, path in ONNX_PATHS.items():
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(
                    f"ONNX-модель не найдена: {path}. "
                    "Запустите src/export_onnx.py перед стартом сервиса."
                )
            self._sessions[model_name] = ort.InferenceSession(
                str(p),
                providers=["CPUExecutionProvider"],
            )
            self._caches[model_name] = LRUCache(MAX_CACHE)
            size_mb = p.stat().st_size / 1024 / 1024
            logger.info("Загружена модель %s: %s (%.1f MB)", model_name, path, size_mb)

        logger.info("WeatherPredictor готов. Модели: %s", list(self._sessions))

    def predict(self, image: Image.Image, model: str = "student") -> dict:
        if model not in self._sessions:
            raise ValueError(
                f"Неизвестная модель '{model}'. Доступны: {list(self._sessions)}"
            )

        # ── Кэш ──────────────────────────────────────────────────────────────
        cache_key = f"{model}:{_image_hash(image)}"
        cached = self._caches[model].get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", cache_key)
            return {**cached, "inference_ms": 0.0, "cached": True}

        # ── Инференс ─────────────────────────────────────────────────────────
        sess = self._sessions[model]
        x = _preprocess(image)

        t0 = time.perf_counter()
        if model == "teacher":
            logits_dn, logits_wt = sess.run(["logits_dn", "logits_wt"], {"image": x})
            probs_dn    = _softmax(logits_dn[0])
            probs_wt    = _softmax(logits_wt[0])
            probs_9     = np.outer(probs_dn, probs_wt).flatten()
            probs_final = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            for (dn, wt), idx in COMBO_TO_FINAL.items():
                probs_final[idx] += probs_9[dn * 5 + wt]
        else:
            logits      = sess.run(["logits"], {"image": x})[0][0]
            probs_final = _softmax(logits)
        inference_ms = (time.perf_counter() - t0) * 1000

        pred_idx   = int(probs_final.argmax())
        confidence = float(probs_final[pred_idx])
        class_name = CLASS_NAMES[pred_idx]
        probs_dict = {name: round(float(p), 6) for name, p in zip(CLASS_NAMES, probs_final)}

        result = {
            "class":         class_name,
            "confidence":    round(confidence, 6),
            "probabilities": probs_dict,
            "model":         model,
            "inference_ms":  round(inference_ms, 2),
            "cached":        False,
        }

        self._caches[model].put(cache_key, result)
        logger.info(
            "predict | model=%s class=%s confidence=%.3f inference_ms=%.1f",
            model, class_name, confidence, inference_ms,
        )
        return result

    def cache_stats(self) -> dict:
        return {name: len(c) for name, c in self._caches.items()}