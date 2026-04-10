import torch
from torchvision import transforms
from PIL import Image
import hydra
from omegaconf import DictConfig

from src.models.weather_model import WeatherModel


def predict(image_path: str, checkpoint_path: str, device: str = "cuda"):
    # Загрузка модели
    model = WeatherModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)

    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Предсказание
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    classes = ['clear', 'fog', 'night', 'night_fog', 'night_rain', 'night_snow', 'rain', 'snow']

    return classes[pred], confidence


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <checkpoint_path>")
        sys.exit(1)

    pred, conf = predict(sys.argv[1], sys.argv[2])
    print(f"Predicted: {pred} (confidence: {conf:.2%})")