import torch
import torch.nn as nn
from torchvision import models


class WeatherModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
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

    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self