# Uncertainty-Aware Deformable DETR (CS776 Project)

> Calibrated reliability for next-generation object detection systems.
>
> We extend Deformable DETR from a "detection-only" model into an
> **uncertainty-aware detector** that separates aleatoric (data) and
> epistemic (model) uncertainty, reports calibrated confidence, and
> supports selective prediction.

**Team:** Maradana Kasi Sri Roshan, Krishna Kumar Bais, Mahathi Garapati,
Daksh Kumar Singh, Sontam Deekshitha, Chinthapudi Gowtham Chand,
Sevak Baliram Shekokar.

**Repository:** <https://github.com/krishna-kumar-bais/Uncertainty-Aware-Deformable-DETR>

## 1. What is inside this repository

- `Track A+B+C/` — main runnable codebase (all three tracks).
  - `Deformable-DETR/` — detector source (model, transformer, datasets, ops).
  - `setup_gpu_server.sh` — one-shot CUDA + Python environment setup.
  - `validate_gpu_env.py` — verifies package versions, CUDA, and ops.
  - `download_checkpoint.py` — downloads official pretrained Deformable DETR.
  - `sweep_mc_dropout.py` — sweeps MC-dropout settings (T, scope, score, temperature).
  - `compare_runs.py` — compares baseline vs. improved runs (AP, ECE, PR-AUC).
  - `predict_images.py` — runs inference on custom images with uncertainty overlays.
- `Report.pdf` — final report.
- `Track A/`, `Track B/` — legacy per-track notebooks (kept for reference).

## 2. Method overview (matches report)

- **Track A – Aleatoric variance head.**
  A parallel MLP `var_embed` predicts per-box `log_var`. Trained with a
  numerically stabilized heteroscedastic L2 objective:
  `L_A = 0.5 * (exp(-log_var) * (b - b_hat)^2 + log_var + log(2π))`.
  By default the detector is frozen and only `var_embed` trains.
- **Track B – Epistemic MC dropout.**
  Keeps decoder FFN dropout active at inference and runs `T` stochastic
  forward passes. Variance across passes becomes the epistemic estimate.
  **Operating point: `T = 15`** (stable epistemic estimate with moderate cost).
- **Track C – Uncertainty-weighted fine-tuning.**
  Per-image training weights are computed from predicted uncertainty,
  normalized to mean 1, and bounded to `[w_min, w_max]`.
- **Calibration & evaluation.**
  We report COCO AP, Expected Calibration Error (ECE), reliability diagrams,
  and precision–recall under uncertainty.

## 3. System requirements

- Linux machine with an NVIDIA GPU and the matching CUDA toolkit (11.8 or 12.1).
- Python **3.10** or **3.11** (the setup script enforces this).
- Disk space for COCO 2017 (~25 GB) and checkpoints (~200 MB).

> You can technically run evaluation on CPU, but custom CUDA ops require GPU.
> On macOS without CUDA you can still read the code and inspect example
> `outputs/*.json` to review the recorded metrics.

## 4. Project layout after setup

```text
Uncertainty-Aware-Deformable-DETR-main/
├── README.md
├── Report.pdf
├── SETUP_COMPLETE.md
├── download_checkpoint.py
├── Track A/
├── Track B/
└── Track A+B+C/
    ├── Deformable-DETR/
    ├── data/
    │   └── coco/
    ├── setup_gpu_server.sh
    ├── validate_gpu_env.py
    ├── download_checkpoint.py
    ├── sweep_mc_dropout.py
    ├── compare_runs.py
    ├── predict_images.py
    ├── GPU_SERVER_SETUP.md
    ├── UNCERTAINTY_EXPERIMENTS.md
    └── build-tools-py38.txt
```

The `outputs/` folder under `Track A+B+C/` is not committed; it is created
automatically when you run the evaluation commands below (via each
command's `--output_dir` flag).

**Where Track C lives (spread across 4 files in `Track A+B+C/Deformable-DETR/`):**

| Role | File | Key location |
| --- | --- | --- |
| Flags defined | `main.py` | `--uncertainty_weighted_loss`, `--uncertainty_weight_source`, `--uncertainty_weight_alpha`, `--uncertainty_weight_topk` |
| Weight computation | `util/uncertainty.py` | `compute_sample_weights(...)` |
| Training hook | `engine.py` | `train_one_epoch(...)` passes `sample_weights` into the criterion |
| Applied to losses | `models/deformable_detr.py` | `loss_labels`, `loss_boxes`, `loss_track_a`, `get_loss`, `forward` |

**All commands below assume you are inside `Track A+B+C/` unless stated otherwise.**

## 5. Step-by-step setup

### 5.1 Clone the repository

```bash
git clone https://github.com/krishna-kumar-bais/Uncertainty-Aware-Deformable-DETR.git
cd Uncertainty-Aware-Deformable-DETR
cd "Track A+B+C"
```

### 5.2 Option A (recommended): automated setup

This builds a `.venv`, installs pinned PyTorch + deps, compiles CUDA ops, runs
the op unit test, and validates the environment.

For CUDA 11.8:

```bash
CUDA_FLAVOR=cu118 bash setup_gpu_server.sh
source .venv/bin/activate
```

For CUDA 12.1:

```bash
CUDA_FLAVOR=cu121 bash setup_gpu_server.sh
source .venv/bin/activate
```

### 5.3 Option B: manual setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

# Match the CUDA your driver supports (cu118 or cu121)
python -m pip install \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r Deformable-DETR/requirements-py38-server.txt

# Build custom CUDA ops
cd Deformable-DETR/models/ops
python setup.py build install
python test.py          # should print "all checking is True"
cd ../../..

# Validate
python validate_gpu_env.py --expect-cuda \
  --requirements Deformable-DETR/requirements-py38-server.txt
```

### 5.4 Prepare COCO 2017

Download from https://cocodataset.org and arrange:

```text
Track A+B+C/data/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

If your COCO lives elsewhere, pass `--coco_path /absolute/path/to/coco`
to every command.

### 5.5 Download the pretrained Deformable DETR checkpoint

```bash
source .venv/bin/activate
python download_checkpoint.py deformable_detr
```

This places the checkpoint at:

```text
Deformable-DETR/models/r50_deformable_detr-checkpoint.pth
```

Other variants are available:

```bash
python download_checkpoint.py deformable_detr_plus   # + iterative box refinement
python download_checkpoint.py deformable_detr_pp     # ++ two-stage
```

## 6. Run the baseline evaluation

```bash
source .venv/bin/activate
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --uncertainty_score confidence \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/base_eval
```

Outputs:

- `outputs/base_eval/eval_metrics.json` — COCO AP + uncertainty summary
- `outputs/base_eval/uncertainty_eval/metrics.json` — ECE, PR-AUC, mean conf/acc
- `outputs/base_eval/uncertainty_eval/reliability_bins.csv`
- `outputs/base_eval/uncertainty_eval/uncertainty_pr.csv`
- `outputs/base_eval/uncertainty_eval/reliability_diagram.png`
- `outputs/base_eval/uncertainty_eval/uncertainty_pr_curve.png`

## 7. Track A — aleatoric variance head

Train only the new variance head (detector frozen):

```bash
python Deformable-DETR/main.py \
  --track_a \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_a
```

To fine-tune detector + variance head jointly:

```bash
python Deformable-DETR/main.py \
  --track_a \
  --no_track_a_freeze_detector \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_a_joint
```

Resulting checkpoint: `outputs/track_a/checkpoint.pth`.

Evaluate Track A with aleatoric-aware uncertainty:

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --uncertainty_score aleatoric \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_a_eval
```

## 8. Track B — MC dropout (operating point T=15)

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --mc_dropout \
  --mc_runs 15 \
  --mc_dropout_scope decoder_ffn \
  --uncertainty_score combined \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/track_b_t15
```

Notes:

- `--mc_dropout_scope` ∈ {`decoder_ffn`, `decoder`, `all`}. `decoder_ffn` is
  cheapest and the setting used in the report.
- `--uncertainty_score` ∈ {`confidence`, `aleatoric`, `epistemic`, `entropy`, `combined`}.

## 9. Track C — uncertainty-weighted fine-tuning

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

Optional knobs:

- `--uncertainty_weight_source aleatoric|epistemic|combined|confidence`
- `--uncertainty_weight_alpha` (strength, default 1.0)
- `--uncertainty_weight_topk` (queries averaged to form per-image weight)
- `--uncertainty_weight_min`, `--uncertainty_weight_max` (clip bounds)

## 10. Full uncertainty-aware evaluation (A + B + C)

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --mc_dropout \
  --mc_runs 15 \
  --mc_dropout_scope decoder_ffn \
  --uncertainty_score combined \
  --confidence_temperature 1.25 \
  --resume outputs/track_c/checkpoint.pth \
  --coco_path data/coco \
  --output_dir outputs/full_eval
```

This is the configuration that matches the final presentation's
"Uncertainty-Aware Model (Full)" row: mAP ≈ 43.3, AP_S ≈ 25.1, ECE ≈ 0.035.

## 11. Compare baseline vs improved

```bash
python compare_runs.py \
  --base outputs/base_eval \
  --improved outputs/full_eval
```

Prints AP, AP50, AP75, AP_S, AP_M, AP_L, ECE (delta convention: positive
delta means improved reduced calibration error), PR-AUC, mean confidence,
mean accuracy.


## 12. Report

The full IEEE-style report is at the repo root: `Report.pdf`.

It covers problem statement, related work, methodology, experimental
setup, results, a detailed section on the **benefits of decreasing ECE**,
and discussion/future work.
