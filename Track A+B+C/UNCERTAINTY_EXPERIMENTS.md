# Uncertainty-Aware Deformable DETR Experiments

This repo implements the project plan as three tracks on top of Deformable DETR.

## Environment

Use the pinned dependencies in [requirements.txt](C:/Users/MyPc/Downloads/CS776_Project/CS776_Project/Deformable-DETR/requirements.txt:1).

CUDA 12.1 example:

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r Deformable-DETR/requirements.txt
```

For a full GPU server workflow, including CUDA-op compilation and validation, use [GPU_SERVER_SETUP.md](C:/Users/MyPc/Downloads/CS776_Project/CS776_Project/GPU_SERVER_SETUP.md:1).

## Track A: Aleatoric Uncertainty Head

Enable a parallel variance head for bounding box regression:

```bash
python Deformable-DETR/main.py \
  --track_a \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_a
```

By default, `--track_a` freezes the detector and trains only `var_embed`.
To fine-tune the detector and variance head together, add:

```bash
--no_track_a_freeze_detector
```

## Track B: Epistemic Uncertainty with MC Dropout

Evaluate with decoder FFN dropout active for stochastic inference:

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --mc_dropout \
  --mc_runs 10 \
  --mc_dropout_scope decoder_ffn \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/mc_eval
```

The research target is not to hardcode `T=10`. Start with lower values such as `2`, `3`, `5`, `8`, and `10`, and keep the smallest `T` that preserves AP while improving ECE and uncertainty PR.

You can automate this search with:

```bash
python sweep_mc_dropout.py \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path data/coco \
  --output_root outputs/mc_sweep
```

This writes `sweep_summary.csv` and `sweep_summary.json` so you can identify the best cost-quality point.

## Track C: Uncertainty-Weighted Fine-Tuning

Use uncertainty to give hard images more training attention:

```bash
python Deformable-DETR/main.py \
  --track_a \
  --no_track_a_freeze_detector \
  --uncertainty_weighted_loss \
  --uncertainty_weight_source aleatoric \
  --uncertainty_weight_alpha 1.0 \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_c
```

The weights are normalized to mean 1 per batch, so this changes emphasis without inflating the overall loss scale.

## Evaluation Outputs

Use the same evaluation framing from the presentation:

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --uncertainty_score combined \
  --resume outputs/track_c/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/final_eval
```

The evaluator reports:

- Average Precision from the standard COCO evaluator.
- Expected Calibration Error (ECE), stored in `outputs/final_eval/uncertainty_eval/metrics.json`.
- Precision-recall under uncertainty, stored in `outputs/final_eval/uncertainty_eval/uncertainty_pr.csv`.
- Reliability diagram bins in `outputs/final_eval/uncertainty_eval/reliability_bins.csv`.
- Reliability diagram image in `outputs/final_eval/uncertainty_eval/reliability_diagram.png` when matplotlib is installed.
- Precision-recall under uncertainty plot in `outputs/final_eval/uncertainty_eval/uncertainty_pr_curve.png` when matplotlib is installed.

## Baseline vs Improved Comparison

Run the baseline deterministically:

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --uncertainty_score confidence \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/base_eval
```

Then compare it with an improved run, such as Track B with `T=10`:

```bash
python compare_runs.py \
  --base outputs/base_eval \
  --improved outputs/mc_eval
```

## Recommended Extra Ablations

- Sweep lower `T` first: `2`, `3`, `5`, `8`, `10`. The best story is usually the smallest good `T`.
- Compare `--uncertainty_score confidence`, `aleatoric`, `epistemic`, `entropy`, and `combined`.
- Compare decoder-only MC dropout with `--mc_dropout_scope decoder` to test cost versus uncertainty quality.
- Tune post-hoc confidence calibration with `--confidence_temperature`; values like `1.0`, `1.25`, and `1.5` are good first checks.
- Report AP drop alongside ECE improvement. The target from the slides is to keep AP within about 1 point while reducing ECE.
