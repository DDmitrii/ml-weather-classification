from pathlib import Path

import torch


def save_model(model_state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
