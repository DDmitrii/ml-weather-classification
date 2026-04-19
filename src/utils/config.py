from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra


def load_config(config_name: str = "train") -> DictConfig:
    """Загрузить конфиг вне Hydra-контекста (для ноутбуков и скриптов)."""
    config_path = str(Path(__file__).parents[2] / "configs")
    with hydra.initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name)
    return cfg


def print_config(cfg: DictConfig) -> None:
    """Красивый вывод всего конфига."""
    print(OmegaConf.to_yaml(cfg))