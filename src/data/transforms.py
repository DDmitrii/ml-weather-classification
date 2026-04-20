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

    resize_size = augmentation_config.get("resize_size", image_size)
    transform_steps = [transforms.Resize((resize_size, resize_size))]

    if train:
        if augmentation_config.get("random_resized_crop", False):
            crop_cfg = augmentation_config["random_resized_crop"]
            transform_steps.append(
                transforms.RandomResizedCrop(
                    image_size,
                    scale=tuple(crop_cfg.get("scale", [0.8, 1.0])),
                    ratio=tuple(crop_cfg.get("ratio", [0.75, 1.3333333333333333])),
                )
            )
        elif resize_size != image_size:
            transform_steps.append(transforms.CenterCrop((image_size, image_size)))
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
        if augmentation_config.get("gaussian_blur"):
            blur_cfg = augmentation_config["gaussian_blur"]
            transform_steps.append(
                transforms.GaussianBlur(
                    kernel_size=blur_cfg.get("kernel_size", 3),
                    sigma=tuple(blur_cfg.get("sigma", [0.1, 2.0])),
                )
            )
        if augmentation_config.get("random_grayscale"):
            transform_steps.append(
                transforms.RandomGrayscale(
                    p=augmentation_config["random_grayscale"].get("p", 0.1)
                )
            )
        if augmentation_config.get("rand_augment"):
            randaug_cfg = augmentation_config["rand_augment"]
            transform_steps.append(
                transforms.RandAugment(
                    num_ops=randaug_cfg.get("num_ops", 2),
                    magnitude=randaug_cfg.get("magnitude", 9),
                )
            )
        if augmentation_config.get("random_erasing"):
            erasing_cfg = augmentation_config["random_erasing"]
        else:
            erasing_cfg = None
    elif resize_size != image_size:
        transform_steps.append(transforms.CenterCrop((image_size, image_size)))

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize["mean"],
                std=normalize["std"],
            ),
        ]
    )

    if train and erasing_cfg:
        transform_steps.append(
            transforms.RandomErasing(
                p=erasing_cfg.get("p", 0.25),
                scale=tuple(erasing_cfg.get("scale", [0.02, 0.12])),
                ratio=tuple(erasing_cfg.get("ratio", [0.3, 3.3])),
            )
        )

    return transforms.Compose(transform_steps)
