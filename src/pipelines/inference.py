import argparse

import torch
from PIL import Image

from src.data.transforms import get_transforms
from src.models.model import create_model
from src.utils.config import load_experiment_config
from src.utils.device import get_device


def predict(image_path, checkpoint_path, config_path="configs/config.yaml", device=None):
    config = load_experiment_config(config_path)
    device = torch.device(device) if device else get_device()

    classes = config["data"].get(
        "class_names",
        [
            "clear",
            "fog",
            "night",
            "night_fog",
            "night_rain",
            "night_snow",
            "rain",
            "snow",
        ],
    )

    model = create_model(
        num_classes=len(classes),
        model_name=config["model"].get("name", "resnet18"),
        pretrained=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = get_transforms(
        image_size=config["data"].get("image_size", 224),
        augmentation_config=config.get("augmentation", {}),
        train=False,
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    return classes[pred], confidence


def main():
    parser = argparse.ArgumentParser(description="Inference for weather classification")
    parser.add_argument("image_path", type=str, help="Path to image")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to experiment config",
    )
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    args = parser.parse_args()

    pred, conf = predict(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        device=args.device,
    )
    print(f"Predicted: {pred} (confidence: {conf:.2%})")


if __name__ == "__main__":
    main()
