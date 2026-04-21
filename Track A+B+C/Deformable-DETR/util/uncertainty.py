# ------------------------------------------------------------------------
# Uncertainty utilities for Deformable DETR research experiments.
# ------------------------------------------------------------------------

import csv
import json
import math
from pathlib import Path

import torch

from util import box_ops
from util.misc import all_gather


def _as_float_tensor(values):
    if len(values) == 0:
        return torch.empty(0, dtype=torch.float32)
    return torch.as_tensor(values, dtype=torch.float32)


def _as_bool_tensor(values):
    if len(values) == 0:
        return torch.empty(0, dtype=torch.bool)
    return torch.as_tensor(values, dtype=torch.bool)


def _normalize_weights(weights, min_weight=0.25, max_weight=4.0):
    weights = weights.float()
    weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(min=min_weight, max=max_weight)
    return weights / weights.mean().clamp_min(1e-6)


def compute_sample_weights(
    outputs,
    source="aleatoric",
    alpha=1.0,
    topk=50,
    min_weight=0.25,
    max_weight=4.0,
):
    """Build per-image training weights from predicted uncertainty.

    The returned weights are normalized to mean 1, so enabling this changes
    which images receive attention without changing the overall loss scale.
    """
    uncertainty_terms = []

    if source in ("aleatoric", "combined") and "pred_log_vars" in outputs:
        aleatoric = outputs["pred_log_vars"].detach().exp().mean(dim=-1)
        uncertainty_terms.append(aleatoric)

    if source in ("epistemic", "combined") and "pred_epistemic_class_var" in outputs:
        epistemic_cls = outputs["pred_epistemic_class_var"].detach().mean(dim=-1)
        uncertainty_terms.append(epistemic_cls)

    if source in ("epistemic", "combined") and "pred_epistemic_box_var" in outputs:
        epistemic_box = outputs["pred_epistemic_box_var"].detach().mean(dim=-1)
        uncertainty_terms.append(epistemic_box)

    if not uncertainty_terms:
        logits = outputs["pred_logits"].detach()
        probs = logits.sigmoid()
        confidence_uncertainty = 1.0 - probs.max(dim=-1).values
        uncertainty_terms.append(confidence_uncertainty)

    query_uncertainty = torch.stack(uncertainty_terms).sum(dim=0)
    k = min(max(int(topk), 1), query_uncertainty.shape[1])
    image_uncertainty = query_uncertainty.topk(k, dim=1).values.mean(dim=1)
    normalized = image_uncertainty / image_uncertainty.mean().clamp_min(1e-6)
    weights = 1.0 + float(alpha) * (normalized - 1.0)
    return _normalize_weights(weights, min_weight=min_weight, max_weight=max_weight)


def _dropout_matches_scope(name, scope):
    if scope == "all":
        return True
    if scope == "decoder":
        return "transformer.decoder" in name
    if scope == "decoder_ffn":
        return "transformer.decoder" in name and (
            name.endswith("dropout3") or name.endswith("dropout4")
        )
    raise ValueError(f"Unknown MC dropout scope: {scope}")


def _set_mc_dropout_modules(model, scope):
    changed_modules = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and _dropout_matches_scope(name, scope):
            changed_modules.append((module, module.training))
            module.train()
    return changed_modules


def _restore_modules(changed_modules):
    for module, was_training in changed_modules:
        module.train(was_training)


def aggregate_mc_outputs(outputs_list):
    """Aggregate stochastic detector outputs from MC dropout runs."""
    logits = torch.stack([out["pred_logits"] for out in outputs_list], dim=0)
    boxes = torch.stack([out["pred_boxes"] for out in outputs_list], dim=0)

    probs = logits.sigmoid()
    mean_probs = probs.mean(dim=0).clamp(min=1e-6, max=1.0 - 1e-6)
    mean_logits = torch.logit(mean_probs)
    mean_boxes = boxes.mean(dim=0).clamp(min=0.0, max=1.0)

    out = {
        "pred_logits": mean_logits,
        "pred_boxes": mean_boxes,
        "pred_epistemic_class_var": probs.var(dim=0, unbiased=False),
        "pred_epistemic_box_var": boxes.var(dim=0, unbiased=False),
    }

    if all("pred_log_vars" in mc_out for mc_out in outputs_list):
        aleatoric_vars = torch.stack(
            [mc_out["pred_log_vars"].exp() for mc_out in outputs_list], dim=0
        )
        mean_aleatoric_vars = aleatoric_vars.mean(dim=0).clamp_min(1e-8)
        out["pred_log_vars"] = mean_aleatoric_vars.log()

    return out


@torch.no_grad()
def mc_dropout_forward(model, samples, runs=10, scope="decoder_ffn"):
    """Run stochastic forward passes with dropout active during evaluation."""
    runs = max(int(runs), 1)
    was_training = model.training
    model.eval()
    changed_modules = _set_mc_dropout_modules(model, scope)
    try:
        outputs_list = [model(samples) for _ in range(runs)]
    finally:
        _restore_modules(changed_modules)
        model.train(was_training)
    return aggregate_mc_outputs(outputs_list)


class DetectionCalibrationEvaluator:
    """Collects calibration and uncertainty quality metrics for detections."""

    def __init__(self, iou_threshold=0.5, num_bins=15, max_detections=100):
        self.iou_threshold = float(iou_threshold)
        self.num_bins = int(num_bins)
        self.max_detections = int(max_detections)
        self.records = []
        self.total_gt = 0

    def update(self, results, targets):
        for result, target in zip(results, targets):
            target_labels = target["labels"].detach().cpu()
            target_boxes = target["boxes"].detach().cpu()
            orig_h, orig_w = target["orig_size"].detach().cpu().tolist()
            target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
            target_boxes = target_boxes * torch.tensor(
                [orig_w, orig_h, orig_w, orig_h], dtype=torch.float32
            )
            self.total_gt += int(target_labels.numel())

            pred_boxes = result["boxes"].detach().cpu()
            pred_labels = result["labels"].detach().cpu()
            pred_scores = result["scores"].detach().cpu()
            pred_uncertainty = result.get("uncertainty")
            if pred_uncertainty is not None:
                pred_uncertainty = pred_uncertainty.detach().cpu()

            if self.max_detections > 0:
                pred_boxes = pred_boxes[: self.max_detections]
                pred_labels = pred_labels[: self.max_detections]
                pred_scores = pred_scores[: self.max_detections]
                if pred_uncertainty is not None:
                    pred_uncertainty = pred_uncertainty[: self.max_detections]

            correct = self._match_predictions(
                pred_boxes, pred_labels, pred_scores, target_boxes, target_labels
            )

            for i in range(pred_scores.numel()):
                record = {
                    "confidence": float(pred_scores[i].item()),
                    "correct": bool(correct[i].item()),
                }
                if pred_uncertainty is not None:
                    record["uncertainty"] = float(pred_uncertainty[i].item())
                self.records.append(record)

    def _match_predictions(self, pred_boxes, pred_labels, pred_scores, target_boxes, target_labels):
        correct = torch.zeros(pred_scores.numel(), dtype=torch.bool)
        if pred_scores.numel() == 0 or target_labels.numel() == 0:
            return correct

        order = torch.argsort(pred_scores, descending=True)
        matched_targets = set()
        ious, _ = box_ops.box_iou(pred_boxes, target_boxes)

        for pred_idx in order.tolist():
            same_class = torch.nonzero(target_labels == pred_labels[pred_idx], as_tuple=False).flatten()
            if same_class.numel() == 0:
                continue
            available = [idx for idx in same_class.tolist() if idx not in matched_targets]
            if not available:
                continue
            candidate_ious = ious[pred_idx, available]
            best_pos = int(candidate_ious.argmax().item())
            best_target = available[best_pos]
            if float(candidate_ious[best_pos].item()) >= self.iou_threshold:
                correct[pred_idx] = True
                matched_targets.add(best_target)

        return correct

    def synchronize_between_processes(self):
        gathered = all_gather({"records": self.records, "total_gt": self.total_gt})
        self.records = []
        self.total_gt = 0
        for item in gathered:
            self.records.extend(item["records"])
            self.total_gt += int(item["total_gt"])

    def summarize(self, output_dir=None):
        confidences = _as_float_tensor([r["confidence"] for r in self.records])
        correct = _as_bool_tensor([r["correct"] for r in self.records])
        uncertainties = _as_float_tensor(
            [r["uncertainty"] for r in self.records if "uncertainty" in r]
        )

        reliability = self._compute_reliability(confidences, correct)
        uncertainty_pr = self._compute_uncertainty_pr(uncertainties, correct)

        stats = {
            "detections": int(confidences.numel()),
            "ground_truths": int(self.total_gt),
            "ece": reliability["ece"],
            "mean_confidence": reliability["mean_confidence"],
            "mean_accuracy": reliability["mean_accuracy"],
            "uncertainty_pr_auc": uncertainty_pr["auc"],
            "uncertainty_available": bool(uncertainties.numel() == correct.numel() and correct.numel() > 0),
        }

        if output_dir is not None:
            self._save_outputs(Path(output_dir), stats, reliability, uncertainty_pr)

        return stats

    def _compute_reliability(self, confidences, correct):
        if confidences.numel() == 0:
            return {
                "ece": 0.0,
                "mean_confidence": 0.0,
                "mean_accuracy": 0.0,
                "bins": [],
            }

        correct_f = correct.float()
        bins = []
        ece = 0.0
        edges = torch.linspace(0.0, 1.0, self.num_bins + 1)
        for bin_idx in range(self.num_bins):
            lo = edges[bin_idx]
            hi = edges[bin_idx + 1]
            if bin_idx == self.num_bins - 1:
                in_bin = (confidences >= lo) & (confidences <= hi)
            else:
                in_bin = (confidences >= lo) & (confidences < hi)
            count = int(in_bin.sum().item())
            if count == 0:
                accuracy = 0.0
                confidence = 0.0
            else:
                accuracy = float(correct_f[in_bin].mean().item())
                confidence = float(confidences[in_bin].mean().item())
                ece += (count / confidences.numel()) * abs(accuracy - confidence)
            bins.append(
                {
                    "bin": bin_idx,
                    "lower": float(lo.item()),
                    "upper": float(hi.item()),
                    "count": count,
                    "accuracy": accuracy,
                    "confidence": confidence,
                }
            )

        return {
            "ece": float(ece),
            "mean_confidence": float(confidences.mean().item()),
            "mean_accuracy": float(correct_f.mean().item()),
            "bins": bins,
        }

    def _compute_uncertainty_pr(self, uncertainties, correct):
        if uncertainties.numel() != correct.numel() or correct.numel() == 0:
            return {"auc": 0.0, "points": []}

        valid = torch.isfinite(uncertainties)
        uncertainties = uncertainties[valid]
        correct = correct[valid]
        if uncertainties.numel() == 0:
            return {"auc": 0.0, "points": []}

        quantiles = torch.linspace(0.0, 1.0, 101)
        thresholds = torch.quantile(uncertainties, quantiles).unique(sorted=True)
        points = []
        for threshold in thresholds:
            keep = uncertainties <= threshold
            kept = int(keep.sum().item())
            if kept == 0:
                precision = 1.0
                recall = 0.0
                true_positives = 0
            else:
                true_positives = int(correct[keep].sum().item())
                precision = true_positives / kept
                recall = true_positives / max(self.total_gt, 1)
            points.append(
                {
                    "threshold": float(threshold.item()),
                    "kept": kept,
                    "precision": float(precision),
                    "recall": float(recall),
                    "true_positives": true_positives,
                }
            )

        auc = 0.0
        if len(points) > 1:
            sorted_points = sorted(points, key=lambda point: point["recall"])
            for left, right in zip(sorted_points[:-1], sorted_points[1:]):
                width = right["recall"] - left["recall"]
                height = 0.5 * (right["precision"] + left["precision"])
                auc += width * height

        return {"auc": float(auc), "points": points}

    def _save_outputs(self, output_dir, stats, reliability, uncertainty_pr):
        out_dir = output_dir / "uncertainty_eval"
        out_dir.mkdir(parents=True, exist_ok=True)

        with (out_dir / "metrics.json").open("w") as f:
            json.dump(stats, f, indent=2)

        with (out_dir / "reliability_bins.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["bin", "lower", "upper", "count", "accuracy", "confidence"]
            )
            writer.writeheader()
            writer.writerows(reliability["bins"])

        with (out_dir / "uncertainty_pr.csv").open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "threshold",
                    "kept",
                    "precision",
                    "recall",
                    "true_positives",
                ],
            )
            writer.writeheader()
            writer.writerows(uncertainty_pr["points"])

        self._save_reliability_plot(out_dir / "reliability_diagram.png", reliability["bins"])
        self._save_uncertainty_pr_plot(out_dir / "uncertainty_pr_curve.png", uncertainty_pr["points"])

    def _save_reliability_plot(self, path, bins):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not bins:
            return

        centers = [(row["lower"] + row["upper"]) / 2.0 for row in bins]
        accuracies = [row["accuracy"] for row in bins]
        width = 1.0 / max(len(bins), 1)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(centers, accuracies, width=width * 0.9, alpha=0.75, label="Accuracy")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title("Reliability Diagram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _save_uncertainty_pr_plot(self, path, points):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not points:
            return

        points = sorted(points, key=lambda point: point["recall"])
        recalls = [point["recall"] for point in points]
        precisions = [point["precision"] for point in points]

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recalls, precisions, linewidth=2.0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall after filtering uncertain detections")
        ax.set_ylabel("Precision after filtering uncertain detections")
        ax.set_title("Precision-Recall Under Uncertainty")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
