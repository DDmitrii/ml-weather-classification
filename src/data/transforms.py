from torchvision import transforms


def get_transforms(image_size=224, augmentation_config=None, train=False):
    augmentation_config = augmentation_config or {}
    normalize = augmentation_config.get(
        "normalize",
        {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )

    transform_steps = [transforms.Resize((image_size, image_size))]

    if train:
        if augmentation_config.get("horizontal_flip", False):
            transform_steps.append(transforms.RandomHorizontalFlip())
        if augmentation_config.get("rotation_degrees", 0):
            transform_steps.append(
                transforms.RandomRotation(augmentation_config["rotation_degrees"])
            )
        if augmentation_config.get("color_jitter"):
            transform_steps.append(
                transforms.ColorJitter(**augmentation_config["color_jitter"])
            )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize["mean"],
                std=normalize["std"],
            ),
        ]
    )

    return transforms.Compose(transform_steps)
