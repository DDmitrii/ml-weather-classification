from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class WeatherDataset(Dataset):
    def __init__(
        self,
        root: str,
        class_names: list[str],
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for cls in self.class_names:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Папка класса не найдена: {cls_dir}")
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    samples.append((img_path, self.class_to_idx[cls]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, label

    def get_class_weights(self) -> torch.Tensor:
        """Вес каждого класса = 1 / count. Для WeightedCrossEntropy."""
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        weights = [1.0 / counts[i] for i in range(len(self.class_names))]
        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / weights.sum() * len(self.class_names)  # нормализация

    def class_distribution(self) -> dict[str, int]:
        from collections import Counter
        counts = Counter(label for _, label in self.samples)
        return {self.class_names[k]: v for k, v in sorted(counts.items())}


def build_dataloaders(cfg) -> tuple[DataLoader, DataLoader, DataLoader]:
    from src.data.transforms import get_train_transforms, get_val_transforms

    train_ds = WeatherDataset(
        root=cfg.data.train_dir,
        class_names=list(cfg.data.class_names),
        transform=get_train_transforms(cfg.data.img_size),
    )
    val_ds = WeatherDataset(
        root=cfg.data.val_dir,
        class_names=list(cfg.data.class_names),
        transform=get_val_transforms(cfg.data.img_size),
    )
    test_ds = WeatherDataset(
        root=cfg.data.test_dir,
        class_names=list(cfg.data.class_names),
        transform=get_val_transforms(cfg.data.img_size),
    )

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )