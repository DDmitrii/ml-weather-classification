import torch
import torch.nn as nn
from torchvision import models


class WeatherModel(nn.Module):
    def __init__(self, num_classes, model_name="resnet18", pretrained=True, dropout=0.0):
        super().__init__()
        self.model = self._build_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            dropout=dropout,
        )

    @staticmethod
    def _with_optional_dropout(in_features, num_classes, dropout):
        if dropout and dropout > 0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
        return nn.Linear(in_features, num_classes)

    def _build_model(self, model_name, pretrained, num_classes, dropout):
        builders = {
            "resnet18": (
                models.resnet18,
                models.ResNet18_Weights.IMAGENET1K_V1,
                "fc",
            ),
            "resnet34": (
                models.resnet34,
                models.ResNet34_Weights.IMAGENET1K_V1,
                "fc",
            ),
            "convnext_tiny": (
                models.convnext_tiny,
                models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
                "classifier",
            ),
            "convnext_base": (
                models.convnext_base,
                models.ConvNeXt_Base_Weights.IMAGENET1K_V1,
                "classifier",
            ),
            "maxvit_t": (
                models.maxvit_t,
                models.MaxVit_T_Weights.IMAGENET1K_V1,
                "classifier",
            ),
        }

        if model_name not in builders:
            raise ValueError(f"Unsupported model: {model_name}")

        builder, default_weights, classifier_attr = builders[model_name]
        weights = default_weights if pretrained else None
        model = builder(weights=weights)

        if classifier_attr == "fc":
            model.fc = self._with_optional_dropout(model.fc.in_features, num_classes, dropout)
            return model

        if classifier_attr == "classifier":
            if hasattr(model.classifier, "__getitem__"):
                last_linear = model.classifier[-1]
                in_features = last_linear.in_features
                layers = list(model.classifier[:-1])
                if dropout and dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
                layers.append(nn.Linear(in_features, num_classes))
                model.classifier = nn.Sequential(*layers)
                return model

        raise ValueError(f"Unsupported classifier structure for model: {model_name}")

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

def create_model(num_classes=8, model_name="resnet18", pretrained=True, dropout=0.0):
    """Factory function to create WeatherModel."""
    return WeatherModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained,
        dropout=dropout,
    )
