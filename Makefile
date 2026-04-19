.PHONY: install train train-exp train-sweep infer mlflow-ui docker-build docker-train docker-train-exp docker-infer docker-mlflow clean

install:
	pip install -r requirements.txt

train:
	python main.py --config configs/config.yaml

train-exp:
	python main.py --config configs/experiments/train.yaml

train-sweep:
	powershell -ExecutionPolicy Bypass -File scripts/run_augmentation_experiments.ps1

infer:
	python src/pipelines/inference.py $(IMAGE_PATH) $(CHECKPOINT_PATH) --config $(CONFIG_PATH)

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns

docker-build:
	docker compose build

docker-train:
	docker compose run --rm train

docker-train-exp:
	docker compose run --rm -e CONFIG_PATH=configs/experiments/train.yaml train

docker-infer:
	docker compose run --rm -e CONFIG_PATH=$(CONFIG_PATH) -e IMAGE_PATH=$(IMAGE_PATH) -e CHECKPOINT_PATH=$(CHECKPOINT_PATH) -e DEVICE=cpu inference

docker-mlflow:
	docker compose up mlflow-ui

clean:
	powershell -Command "Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force"
	powershell -Command "Get-ChildItem -Recurse -File -Include *.pyc | Remove-Item -Force"
