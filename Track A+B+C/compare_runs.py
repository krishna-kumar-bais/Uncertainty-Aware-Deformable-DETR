#!/usr/bin/env python3
"""
Compare baseline and improved evaluation outputs.
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def collect_metrics(run_dir):
    run_dir = Path(run_dir)
    eval_metrics = load_json(run_dir / "eval_metrics.json")
    uncertainty_metrics = load_json(run_dir / "uncertainty_eval" / "metrics.json")

    metrics = {}
    bbox = eval_metrics.get("coco_eval_bbox")
    if isinstance(bbox, list) and len(bbox) >= 6:
        metrics["AP"] = bbox[0]
        metrics["AP50"] = bbox[1]
        metrics["AP75"] = bbox[2]
        metrics["AP_S"] = bbox[3]
        metrics["AP_M"] = bbox[4]
        metrics["AP_L"] = bbox[5]

    if "ece" in eval_metrics:
        metrics["ECE"] = eval_metrics["ece"]
    elif "ece" in uncertainty_metrics:
        metrics["ECE"] = uncertainty_metrics["ece"]

    if "uncertainty_pr_auc" in eval_metrics:
        metrics["Uncertainty_PR_AUC"] = eval_metrics["uncertainty_pr_auc"]
    elif "uncertainty_pr_auc" in uncertainty_metrics:
        metrics["Uncertainty_PR_AUC"] = uncertainty_metrics["uncertainty_pr_auc"]

    if "mean_confidence" in uncertainty_metrics:
        metrics["Mean_Confidence"] = uncertainty_metrics["mean_confidence"]
    if "mean_accuracy" in uncertainty_metrics:
        metrics["Mean_Accuracy"] = uncertainty_metrics["mean_accuracy"]

    return metrics


def fmt(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def delta_for_metric(metric_name, base_value, improved_value):
    if base_value is None or improved_value is None:
        return None
    delta = improved_value - base_value
    if metric_name == "ECE":
        delta = base_value - improved_value
    return delta


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and improved model runs.")
    parser.add_argument("--base", required=True, help="Baseline run directory")
    parser.add_argument("--improved", required=True, help="Improved run directory")
    args = parser.parse_args()

    base_metrics = collect_metrics(args.base)
    improved_metrics = collect_metrics(args.improved)

    metric_names = [
        "AP",
        "AP50",
        "AP75",
        "AP_S",
        "AP_M",
        "AP_L",
        "ECE",
        "Uncertainty_PR_AUC",
        "Mean_Confidence",
        "Mean_Accuracy",
    ]

    print(f"Base run:      {Path(args.base).resolve()}")
    print(f"Improved run:  {Path(args.improved).resolve()}")
    print()
    print(f"{'Metric':<20} {'Base':>12} {'Improved':>12} {'Delta':>12}")
    print("-" * 58)
    for metric_name in metric_names:
        base_value = base_metrics.get(metric_name)
        improved_value = improved_metrics.get(metric_name)
        delta = delta_for_metric(metric_name, base_value, improved_value)
        print(
            f"{metric_name:<20} {fmt(base_value):>12} {fmt(improved_value):>12} {fmt(delta):>12}"
        )

    print()
    print("Delta convention:")
    print("  - For AP-style metrics: positive is better.")
    print("  - For ECE: positive means the improved model reduced calibration error.")
    print("  - For Uncertainty_PR_AUC: positive is better.")


if __name__ == "__main__":
    main()
