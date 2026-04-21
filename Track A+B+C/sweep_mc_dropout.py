#!/usr/bin/env python3
"""
Sweep Track B MC-dropout settings and save a comparable summary table.
"""

import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep MC-dropout hyperparameters for Deformable DETR.")
    parser.add_argument("--main", default="Deformable-DETR/main.py", help="Path to main.py")
    parser.add_argument("--resume", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--coco_path", required=True, help="Path to the COCO dataset root")
    parser.add_argument("--output_root", required=True, help="Directory to store sweep outputs")
    parser.add_argument("--device", default="cuda", help="Device passed to main.py")
    parser.add_argument("--batch_size", default="2", help="Batch size for evaluation")
    parser.add_argument("--mc_runs", nargs="+", type=int, default=[2, 3, 5, 8, 10],
                        help="MC-dropout sample counts to sweep")
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.0, 1.25, 1.5],
                        help="Confidence temperature values to sweep")
    parser.add_argument("--uncertainty_scores", nargs="+", default=["combined", "entropy"],
                        help="Uncertainty scoring modes to sweep")
    parser.add_argument("--scopes", nargs="+", default=["decoder_ffn"],
                        help="Dropout scopes to sweep")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch main.py")
    parser.add_argument("--extra_args", nargs=argparse.REMAINDER, default=[],
                        help="Extra arguments forwarded to main.py")
    return parser.parse_args()


def load_json(path):
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def summarize_run(run_dir):
    eval_metrics = load_json(run_dir / "eval_metrics.json")
    uncertainty_metrics = load_json(run_dir / "uncertainty_eval" / "metrics.json")
    bbox = eval_metrics.get("coco_eval_bbox")
    summary = {
        "ap": None,
        "ap50": None,
        "ap75": None,
        "ece": eval_metrics.get("ece", uncertainty_metrics.get("ece")),
        "uncertainty_pr_auc": eval_metrics.get(
            "uncertainty_pr_auc", uncertainty_metrics.get("uncertainty_pr_auc")
        ),
        "mean_confidence": uncertainty_metrics.get("mean_confidence"),
        "mean_accuracy": uncertainty_metrics.get("mean_accuracy"),
    }
    if isinstance(bbox, list) and len(bbox) >= 3:
        summary["ap"] = bbox[0]
        summary["ap50"] = bbox[1]
        summary["ap75"] = bbox[2]
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    grid = itertools.product(args.mc_runs, args.temperatures, args.uncertainty_scores, args.scopes)
    for mc_runs, temperature, uncertainty_score, scope in grid:
        run_name = f"T{mc_runs}_temp{temperature:.2f}_{uncertainty_score}_{scope}"
        run_dir = output_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        command = [
            args.python,
            args.main,
            "--eval",
            "--eval_uncertainty",
            "--mc_dropout",
            "--mc_runs", str(mc_runs),
            "--mc_dropout_scope", scope,
            "--uncertainty_score", uncertainty_score,
            "--confidence_temperature", str(temperature),
            "--resume", args.resume,
            "--coco_path", args.coco_path,
            "--output_dir", str(run_dir),
            "--device", args.device,
            "--batch_size", str(args.batch_size),
        ] + args.extra_args

        print(f"\n=== Running {run_name} ===")
        start = time.time()
        completed = subprocess.run(command, check=False)
        elapsed = time.time() - start

        row = {
            "run_name": run_name,
            "mc_runs": mc_runs,
            "temperature": temperature,
            "uncertainty_score": uncertainty_score,
            "scope": scope,
            "elapsed_seconds": round(elapsed, 3),
            "return_code": completed.returncode,
        }
        row.update(summarize_run(run_dir))
        rows.append(row)

    summary_json = output_root / "sweep_summary.json"
    with summary_json.open("w") as f:
        json.dump(rows, f, indent=2)

    summary_csv = output_root / "sweep_summary.csv"
    fieldnames = [
        "run_name",
        "mc_runs",
        "temperature",
        "uncertainty_score",
        "scope",
        "elapsed_seconds",
        "return_code",
        "ap",
        "ap50",
        "ap75",
        "ece",
        "uncertainty_pr_auc",
        "mean_confidence",
        "mean_accuracy",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    good_rows = [row for row in rows if row["return_code"] == 0]
    if not good_rows:
        print("\nNo successful sweep runs completed.")
        return

    good_rows.sort(
        key=lambda row: (
            row["ece"] if row["ece"] is not None else float("inf"),
            -(row["ap"] if row["ap"] is not None else -1.0),
            row["elapsed_seconds"],
        )
    )
    print("\nBest sweep result by low ECE, then high AP, then low runtime:")
    print(json.dumps(good_rows[0], indent=2))


if __name__ == "__main__":
    main()
