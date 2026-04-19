import torch
import torch.nn as nn
from torchvision import models


class WeatherModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", pretrained=True):
        super().__init__()
        if model_name != "resnet18":
            raise ValueError(f"Unsupported model: {model_name}")

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, preds = torch.max(outputs, 1)
        return preds

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        self.to(device)
        return self

def create_model(num_classes=8, model_name="resnet18", pretrained=True):
    """Factory function to create WeatherModel"""
    return WeatherModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
    )
