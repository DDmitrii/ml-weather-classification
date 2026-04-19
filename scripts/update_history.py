import argparse
import json
from pathlib import Path


def load_summary(summary_path):
    with open(summary_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_metric(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def build_history_entry(summary):
    mlflow_info = summary.get("mlflow", {})
    test_metrics = summary.get("test_metrics", {})
    artifacts = summary.get("artifacts", {})
    skipped_report = artifacts.get("skipped_files_report")

    lines = [
        f"## {summary.get('run_name', 'unknown_run')}",
        "",
        f"- Experiment dir: `{artifacts.get('experiment_dir', '-')}`",
        f"- Resolved config: `{artifacts.get('resolved_config', '-')}`",
        f"- Checkpoint: `{artifacts.get('checkpoint', '-')}`",
        f"- Best epoch: `{summary.get('best_epoch', '-')}`",
        f"- Best val accuracy: {format_metric(summary.get('best_val_accuracy'))}",
        f"- Test accuracy: {format_metric(test_metrics.get('test_accuracy'))}",
        f"- Test precision: {format_metric(test_metrics.get('test_precision'))}",
        f"- Test recall: {format_metric(test_metrics.get('test_recall'))}",
        f"- Test F1: {format_metric(test_metrics.get('test_f1'))}",
        f"- MLflow experiment: `{mlflow_info.get('experiment_name', '-')}`",
        f"- MLflow run id: `{mlflow_info.get('run_id', '-')}`",
        f"- MLflow page: {mlflow_info.get('ui_url', '-')}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Append experiment summary to HISTORY.md")
    parser.add_argument("--summary", required=True, help="Path to summary.json")
    parser.add_argument(
        "--history",
        default="HISTORY.md",
        help="Path to HISTORY.md",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    history_path = Path(args.history)

    summary = load_summary(summary_path)
    entry = build_history_entry(summary)

    existing = ""
    if history_path.exists():
        existing = history_path.read_text(encoding="utf-8").rstrip() + "\n\n"

    history_path.write_text(existing + entry + "\n", encoding="utf-8")
    print(f"Updated {history_path} with {summary.get('run_name', 'unknown_run')}")


if __name__ == "__main__":
    main()
