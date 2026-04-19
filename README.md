# ML Weather Classification

Классификация погодных условий по изображениям с логированием экспериментов в MLflow.

## Структура

- `main.py` — точка входа для обучения.
- `configs/` — базовые конфиги данных, модели, обучения и аугментаций.
- `configs/experiments/` — отдельные конфиги экспериментов.
- `src/pipelines/train.py` — обучение, валидация, тест, сохранение артефактов.
- `src/pipelines/inference.py` — инференс по одному изображению.
- `mlruns/` — хранилище MLflow и локальных артефактов запусков.
- `HISTORY.md` — журнал экспериментов.

## Установка

```bash
pip install -r requirements.txt
```

## Обучение

Базовый запуск:

```bash
python main.py --config configs/config.yaml
```

Отдельный эксперимент:

```bash
python main.py --config configs/experiments/train_exp1_hflip_light.yaml
```

Базовый экспериментальный профиль:

```bash
python main.py --config configs/experiments/train.yaml
```

Серия аугментационных экспериментов:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_augmentation_experiments.ps1
```

После запуска результаты лежат в:

- `mlruns/<experiment_id>/<run_id>/...` — внутренние файлы MLflow
- `mlruns/local_artifacts/experiments/<timestamp>_<run_name>/summary.json`
- `mlruns/local_artifacts/experiments/<timestamp>_<run_name>/resolved_config.yaml`
- `mlruns/local_artifacts/experiments/<timestamp>_<run_name>/checkpoints/`
- `mlruns/local_artifacts/experiments/<timestamp>_<run_name>/skipped_files.csv` — если были битые изображения

## MLflow

Запуск UI:

```bash
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000
```

Интерфейс будет доступен по адресу [http://127.0.0.1:5000](http://127.0.0.1:5000).

Обновление `HISTORY.md` по последнему `summary.json`:

```bash
python scripts/update_history.py --summary mlruns/local_artifacts/experiments/<timestamp>_<run_name>/summary.json
```

## Инференс

Локальный запуск:

```bash
python src/pipelines/inference.py path/to/image.png mlruns/local_artifacts/experiments/<run>/checkpoints/best_model_state.pth --config configs/config.yaml --device cpu
```

Где:

- `path/to/image.png` — изображение для предсказания
- `best_model_state.pth` — чекпоинт из нужного запуска
- `--config` — конфиг, совместимый с этим чекпоинтом

## Docker

Сборка:

```bash
docker compose build
```

Обучение в контейнере с базовым конфигом:

```bash
docker compose run --rm train
```

Обучение с экспериментальным конфигом:

```bash
docker compose run --rm -e CONFIG_PATH=configs/experiments/train_exp1_hflip_light.yaml train
```

Инференс в контейнере:

```bash
docker compose run --rm -e CONFIG_PATH=configs/config.yaml -e IMAGE_PATH=path/to/image.png -e CHECKPOINT_PATH=mlruns/local_artifacts/experiments/<run>/checkpoints/best_model_state.pth -e DEVICE=cpu inference
```

MLflow UI в контейнере:

```bash
docker compose up mlflow-ui
```

## Makefile

Упрощённые команды:

```bash
make train
make train-exp
make train-sweep
make mlflow-ui
make docker-build
make docker-train
```
