from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml


SECTION_FILES = {
    "data": "data.yaml",
    "model": "model.yaml",
    "training": "training.yaml",
    "augmentation": "augmentation.yaml",
}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")

    return data


def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _normalize_legacy_config(config: dict) -> dict:
    normalized = deepcopy(config)

    if "data" not in normalized:
        normalized["data"] = {}
    if "model" not in normalized:
        normalized["model"] = {}
    if "training" not in normalized:
        normalized["training"] = {}
    if "augmentation" not in normalized:
        normalized["augmentation"] = {}
    if "mlflow" not in normalized:
        normalized["mlflow"] = {}

    legacy_to_section = {
        "data_dir": ("data", "data_dir"),
        "batch_size": ("data", "batch_size"),
        "num_workers": ("data", "num_workers"),
        "image_size": ("data", "image_size"),
        "learning_rate": ("training", "learning_rate"),
        "num_epochs": ("training", "epochs"),
        "use_scheduler": ("training", "use_scheduler"),
        "experiment_name": ("mlflow", "experiment_name"),
    }

    for old_key, (section, new_key) in legacy_to_section.items():
        if old_key in normalized and new_key not in normalized[section]:
            normalized[section][new_key] = normalized.pop(old_key)

    return normalized


def _find_project_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    return start_path


def _resolve_tracking_uri(tracking_uri: str, project_root: Path) -> str:
    if not tracking_uri:
        return tracking_uri

    if tracking_uri.startswith(("http://", "https://")):
        return tracking_uri

    if tracking_uri.startswith("file:"):
        raw_path = tracking_uri.removeprefix("file:")
        if raw_path.startswith("///"):
            return tracking_uri

        path = Path(raw_path)
    else:
        path = Path(tracking_uri)

    if not path.is_absolute():
        path = (project_root / path).resolve()

    return path.as_uri()


def tracking_uri_to_path(tracking_uri: str) -> Path | None:
    if not tracking_uri:
        return None

    if tracking_uri.startswith(("http://", "https://", "sqlite:")):
        return None

    if tracking_uri.startswith("file:///"):
        parsed = urlparse(tracking_uri)
        parsed_path = unquote(parsed.path)
        if parsed_path.startswith("/") and len(parsed_path) > 2 and parsed_path[2] == ":":
            parsed_path = parsed_path[1:]
        return Path(parsed_path)

    if tracking_uri.startswith("file:"):
        return Path(tracking_uri.removeprefix("file:"))

    return Path(tracking_uri)


def load_experiment_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    config = _read_yaml(config_path)
    config_dir = config_path.parent
    project_root = _find_project_root(config_dir)
    configs_root = project_root / "configs"

    for section, file_name in SECTION_FILES.items():
        section_default = _read_yaml(configs_root / file_name)
        if section not in config or not isinstance(config.get(section), dict) or not config[section]:
            config[section] = _deep_merge(
                section_default,
                config.get(section, {}),
            )
        else:
            config[section] = _deep_merge(section_default, config[section])

    config = _normalize_legacy_config(config)

    config.setdefault("project_name", "ml_weather")
    config.setdefault("seed", 42)
    config.setdefault("save_best_model", True)
    config.setdefault("config_path", str(config_path))
    config["data"].setdefault("data_dir", "data/raw/weather_dataset_complete/weather_dataset_complete")
    config["data"].setdefault("batch_size", 32)
    config["data"].setdefault("num_workers", 2)
    config["data"].setdefault("image_size", 224)
    config["model"].setdefault("name", "resnet18")
    config["model"].setdefault("pretrained", True)
    config["training"].setdefault("epochs", 10)
    config["training"].setdefault("learning_rate", 1e-4)
    config["training"].setdefault("optimizer", "adam")
    config["training"].setdefault("weight_decay", 0.0)
    config["training"].setdefault("use_scheduler", False)
    config["mlflow"].setdefault("enabled", True)
    config["mlflow"].setdefault("tracking_uri", "file:./mlruns")
    config["mlflow"].setdefault("experiment_name", "weather_classification")
    config["mlflow"].setdefault("run_name", "baseline")
    config["mlflow"].setdefault("ui_url", "http://127.0.0.1:5000")
    config["mlflow"].setdefault("tags", {})
    config["mlflow"]["tracking_uri"] = _resolve_tracking_uri(
        config["mlflow"]["tracking_uri"],
        project_root,
    )

    return config


def flatten_config(data: dict, prefix: str = "") -> dict:
    items = {}
    for key, value in data.items():
        compound_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_config(value, compound_key))
        else:
            items[compound_key] = value
    return items
