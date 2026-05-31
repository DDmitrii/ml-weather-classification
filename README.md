# weather-classification

## Задача

Определение погодных условий по снимку, полученному на автомобильной дороге. 
Задача решается как многоклассовая классификация изображений (9 классов).

---

## Классы

| ID | Класс        | Описание      |
|----|--------------|---------------|
| 0 | `clear`      | Ясно + день   |
| 1 | `fog`        | Туман + день  |
| 2 | `fog_rain`   | Туман + дождь |
| 3 | `night`      | Ночь + ясно   |
| 4 | `night_fog`  | Ночь + туман  |
| 5 | `night_rain` | Ночь + дождь  |
| 6 | `night_snow` | Ночь + снег   |
| 7 | `rain`       | Дождь + день  |
| 8 | `snow`       | Снег + день   |

---

## Архитектура модели

Использована **двухголовая архитектура (MultiHead)** поверх backbone ConvNeXt-Tiny. Вместо одного классификатора на 9 классов применяются два параллельных классификатора:

```
Backbone (ConvNeXt-Tiny, pretrained, feat_dim=768)
              ↓
    ┌─────────┴──────────┐
 head_dn (2)         head_wt (5)
 Day / Night      Clear / Fog / Fog_rain / Rain / Snow
    └─────────┬──────────┘
       COMBO_TO_FINAL[dn, wt]
              ↓
    Финальный класс (0–8)
```

Разделение задачи на два ортогональных подпространства 
(день/ночь и тип погоды) снижает путаницу между визуально схожими классами 
(например, `night` и `night_fog`). 
Итоговые вероятности по 9 классам вычисляются через внешнее произведение softmax-ов двух голов.

### Маппинг комбинаций

```python
COMBO_TO_FINAL = {
    (0, 0): 0,  # day + clear   → clear
    (0, 1): 1,  # day + fog     → fog
    (0, 2): 2,  # day + fog_rain → fog_rain
    (0, 3): 7,  # day + rain    → rain
    (0, 4): 8,  # day + snow    → snow
    (1, 0): 3,  # night + clear → night
    (1, 1): 4,  # night + fog   → night_fog
    (1, 2): 2,  # night + fog_rain → fog_rain (edge case)
    (1, 3): 5,  # night + rain  → night_rain
    (1, 4): 6,  # night + snow  → night_snow
}
```

### Функция потерь

Применяется **Focal Loss** с `γ = 2.0` для борьбы с дисбалансом классов. Итоговый лосс — сумма потерь двух голов:

```
L = FocalLoss(logits_dn, y_dn) + λ · FocalLoss(logits_wt, y_wt),  λ = 1.0
```

---

## Старт

### Установка

```bash
git clone https://github.com/<DDmitrii>/ml-weather-classification

pip install -r requirements.txt
```

### Обучение

```bash
# Лучший конфиг (Focal Loss MultiHead)
python -m src.train_pipeline experiments=focal_loss_multihead

# Baseline
python -m src.train_pipeline experiments=baseline_convnext_tiny
```

### Модели

| Модель | Backbone | Размер | Accuracy   | Латентность |
|--------|----------|--------|------------|-------------|
| `teacher` | ConvNeXt-Tiny (MultiHead) | 106 MB | **94.71%** | ~25 ms |
| `student` | MobileNetV3-Small | 5.8 MB | 92.31%     | ~5 ms |

---

## Датасет

**Скачать:** https://www.kaggle.com/datasets/ffftory/merged-weather-v7

Используются два публичных датасета изображений с камер наблюдения и дорожных камер:

- **ACDC** — Adverse Conditions Dataset with Correspondences
- **DAWN** — Detection in Adverse Weather Nature
- Ручной парсинг интернета
- Подключение автономного ИИ агента для автоматизации поиска необходимых изображений

Изображения ресайзятся до 224×224 и нормализуются по ImageNet-статистике (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

**Аугментации при обучении:** горизонтальное отражение, цветовые искажения, размытие, аффинные трансформации, гамма-коррекция. **WeightedRandomSampler** активен всегда — балансировка классов встроена в загрузчик данных.

---

## Зависимости

- Python 3.11, PyTorch 2.x, PyTorch Lightning
- timm (backbone zoo), ONNX Runtime, FastAPI, Pillow
- Hydra-core, MLflow
