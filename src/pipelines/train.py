import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import mlflow
import torch.nn as nn
import torch.optim as optim
import yaml

from src.data.dataloader import (
    collect_skipped_files,
    create_dataloaders,
    write_skipped_files_report,
)
from src.models.evaluate import evaluate
from src.models.model import create_model
from src.models.train import train
from src.utils.config import tracking_uri_to_path
from src.utils.device import get_device
from src.utils.metrics import compute_metrics
from src.utils.model_io import save_model
from src.utils.seed import set_seed


def _build_optimizer(model, training_config):
    optimizer_name = training_config.get("optimizer", "adam").lower()
    learning_rate = training_config["learning_rate"]
    weight_decay = training_config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if optimizer_name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=training_config.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")


def _create_experiment_dir(config):
    tracking_root = tracking_uri_to_path(config["mlflow"].get("tracking_uri", ""))
    output_dir = tracking_root / "local_artifacts" if tracking_root else Path("mlruns") / "local_artifacts"
    run_name = config["mlflow"].get("run_name", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_dir / "experiments" / f"{timestamp}_{run_name}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def run_training_pipeline(config, use_mlflow=True):
    """Main training pipeline with shared config and experiment tracking."""
    set_seed(config.get("seed", 42))
    device = get_device()
    experiment_dir = _create_experiment_dir(config)

    if use_mlflow:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    print(f"Using device: {device}")
    print(f"MLflow logging: {'Enabled' if use_mlflow else 'Disabled'}")
    print(f"Experiment directory: {experiment_dir}")

    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir=data_config["data_dir"],
        batch_size=data_config["batch_size"],
        num_workers=data_config.get("num_workers", 2),
        image_size=data_config.get("image_size", 224),
        augmentation_config=config.get("augmentation", {}),
    )

    print("\nDataset info:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Classes: {classes}")

    skipped_files_path = experiment_dir / "skipped_files.csv"

    model = create_model(
        num_classes=len(classes),
        model_name=model_config.get("name", "resnet18"),
        pretrained=model_config.get("pretrained", True),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, training_config)
    scheduler = None
    if training_config.get("use_scheduler", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=training_config.get("scheduler_patience", 3),
            factor=training_config.get("scheduler_factor", 0.5),
        )

    resolved_config_path = experiment_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    checkpoint_path = experiment_dir / "checkpoints" / "best_model_state.pth"
    test_metrics = {}
    mlflow_info = {
        "enabled": use_mlflow,
        "tracking_uri": config["mlflow"].get("tracking_uri"),
        "experiment_name": config["mlflow"].get("experiment_name"),
        "run_name": config["mlflow"].get("run_name"),
        "ui_url": None,
        "experiment_id": None,
        "run_id": None,
    }

    run_context = nullcontext()
    if use_mlflow:
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        run_context = mlflow.start_run(run_name=config["mlflow"].get("run_name"))

    with run_context:
        if use_mlflow:
            active_run = mlflow.active_run()
            if active_run is not None:
                mlflow_info["run_id"] = active_run.info.run_id
                mlflow_info["experiment_id"] = active_run.info.experiment_id
                mlflow_info["ui_url"] = (
                    f"{config['mlflow'].get('ui_url', 'http://127.0.0.1:5000')}"
                    f"/#/experiments/{active_run.info.experiment_id}/runs/{active_run.info.run_id}"
                )

        best_model_state, best_val_acc, best_epoch, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            config=config,
            scheduler=scheduler,
            use_mlflow=use_mlflow,
        )

        if best_model_state is not None:
            save_model(best_model_state, checkpoint_path)
            model.load_state_dict(best_model_state)
            model.save(experiment_dir / "checkpoints" / "best_model_weather.pth")

            test_loss, test_acc, test_preds, test_labels = evaluate(
                model, test_loader, criterion, device
            )
            test_extra_metrics = compute_metrics(test_labels, test_preds)
            test_metrics = {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_precision": test_extra_metrics["precision"],
                "test_recall": test_extra_metrics["recall"],
                "test_f1": test_extra_metrics["f1"],
            }

            print("\nTest Results:")
            for metric_name, metric_value in test_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")

            if use_mlflow:
                mlflow.log_artifact(str(resolved_config_path))
                mlflow.log_artifact(str(checkpoint_path))
                mlflow.log_metrics(test_metrics)

        skipped_files = collect_skipped_files(
            train_loader.dataset,
            val_loader.dataset,
            test_loader.dataset,
        )
        if skipped_files:
            write_skipped_files_report(skipped_files_path, skipped_files)
            print(f"\nSkipped corrupted files: {len(skipped_files)}")
            print(f"Skipped files report: {skipped_files_path}")
            if use_mlflow:
                mlflow.log_artifact(str(skipped_files_path))

    summary = {
        "project_name": config.get("project_name", "ml_weather"),
        "run_name": config["mlflow"].get("run_name", "baseline"),
        "classes": classes,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "history": history,
        "test_metrics": test_metrics,
        "mlflow": mlflow_info,
        "artifacts": {
            "experiment_dir": str(experiment_dir),
            "resolved_config": str(resolved_config_path),
            "checkpoint": str(checkpoint_path),
            "skipped_files_report": str(skipped_files_path) if skipped_files_path.exists() else None,
        },
    }
    summary_path = experiment_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    return best_model_state, best_val_acc
