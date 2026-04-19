import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torchvision import datasets

from src.data.transforms import get_transforms


class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, *args, split_name="dataset", **kwargs):
        super().__init__(*args, **kwargs)
        self.split_name = split_name
        self.skipped_files = []

    def __getitem__(self, index):
        path, _ = self.samples[index]
        try:
            return super().__getitem__(index)
        except Exception as exc:
            self.skipped_files.append(
                {
                    "split": self.split_name,
                    "path": path,
                    "error": str(exc),
                }
            )
            return None


def safe_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return default_collate(batch)


def _build_loader(dataset, batch_size, shuffle, pin_memory, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        collate_fn=safe_collate,
    )


def create_dataloaders(
    data_dir,
    batch_size=64,
    num_workers=2,
    image_size=224,
    augmentation_config=None,
):
    """Create train, val, test dataloaders."""
    train_transform = get_transforms(
        image_size=image_size,
        augmentation_config=augmentation_config,
        train=True,
    )
    eval_transform = get_transforms(
        image_size=image_size,
        augmentation_config=augmentation_config,
        train=False,
    )

    train_dataset = SafeImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform,
        split_name="train",
    )
    val_dataset = SafeImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=eval_transform,
        split_name="val",
    )
    test_dataset = SafeImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=eval_transform,
        split_name="test",
    )

    pin_memory = torch.cuda.is_available()

    train_loader = _build_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    val_loader = _build_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def collect_skipped_files(*datasets_to_check):
    skipped_files = []
    for dataset in datasets_to_check:
        skipped_files.extend(dataset.skipped_files)
    return skipped_files


def write_skipped_files_report(output_path, skipped_files):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["split,path,error"]
    for item in skipped_files:
        safe_error = item["error"].replace('"', "'")
        lines.append(f'{item["split"]},"{item["path"]}","{safe_error}"')

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
