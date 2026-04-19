from copy import deepcopy

import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.models.evaluate import evaluate
from src.utils.config import flatten_config
from src.utils.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Training Epoch {epoch}")
    for batch in pbar:
        if batch is None:
            continue

        inputs, labels = batch
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / max(len(all_labels), 1)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    config,
    scheduler=None,
    use_mlflow=True,
):
    num_epochs = config["training"]["epochs"]
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    history = []

    def run_loop():
        nonlocal best_val_acc, best_model_state, best_epoch

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch + 1
            )
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            val_loss, val_acc, val_preds, val_labels = evaluate(
                model, val_loader, criterion, device
            )
            val_extra_metrics = compute_metrics(val_labels, val_preds)
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            epoch_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_extra_metrics["precision"],
                "val_recall": val_extra_metrics["recall"],
                "val_f1": val_extra_metrics["f1"],
            }
            history.append({"epoch": epoch + 1, **epoch_metrics})

            if use_mlflow:
                mlflow.log_metrics(epoch_metrics, step=epoch)

            if scheduler is not None:
                scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch + 1
                print(f"New best model! Val Acc: {val_acc:.4f}")

                if use_mlflow:
                    mlflow.log_metric("best_val_accuracy", best_val_acc, step=epoch)
                    mlflow.log_metric("best_epoch", best_epoch)

    if use_mlflow:
        mlflow.log_params(
            {key: str(value) for key, value in flatten_config(config).items()}
        )
        if config["mlflow"].get("tags"):
            mlflow.set_tags(config["mlflow"]["tags"])

    run_loop()

    if use_mlflow and best_model_state is not None:
        model.load_state_dict(best_model_state)
        mlflow.pytorch.log_model(model, "best_model")

    print(
        f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}"
    )
    return best_model_state, best_val_acc, best_epoch, history
