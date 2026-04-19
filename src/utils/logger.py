import mlflow
from omegaconf import DictConfig, OmegaConf


class ExperimentLogger:
    """Обёртка над MLflow для логирования экспериментов."""

    def __init__(self, cfg: DictConfig):
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment_name)
        self.cfg = cfg
        self._run = None

    def start(self, run_name: str = None):
        """Начать новый run."""
        self._run = mlflow.start_run(run_name=run_name)
        # Логируем весь конфиг как параметры
        flat = self._flatten(OmegaConf.to_container(self.cfg, resolve=True))
        mlflow.log_params(flat)
        return self

    def log_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def log_model(self, model, artifact_path: str = "model"):
        import torch
        mlflow.pytorch.log_model(model, artifact_path)

    def finish(self):
        mlflow.end_run()

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.finish()

    @staticmethod
    def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Превратить вложенный dict в плоский для mlflow.log_params."""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(ExperimentLogger._flatten(v, new_key, sep))
            elif isinstance(v, list):
                items[new_key] = str(v)
            else:
                items[new_key] = v
        return items