import argparse
from src.pipelines.train import run_training_pipeline
from src.utils.config import load_experiment_config


def main():
    parser = argparse.ArgumentParser(description='Train weather classification model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging')
    args = parser.parse_args()

    # Load config
    config = load_experiment_config(args.config)

    # Run training
    best_model, best_acc = run_training_pipeline(
        config,
        use_mlflow=(not args.no_mlflow) and config["mlflow"].get("enabled", True)
    )

    print(f" Training finished!")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    main()
