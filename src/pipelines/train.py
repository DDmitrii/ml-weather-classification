import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from src.model.model import WeatherModel
from src.model.train import train_one_epoch
from src.model.evaluate import evaluate

from src.utils.device import get_device
from src.utils.seed import set_seed
from src.utils.model_io import save_model


def get_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def run_train():
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = "data/raw/weather_dataset_complete"
    batch_size = 64
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 8

    # 🔹 3. данные
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir, batch_size
    )


    model = WeatherModel(num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, "best_model.pth")
            print(f"✅ Saved new best model (val_acc={val_acc:.4f})")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")

    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")


if __name__ == "__main__":
    run_train()