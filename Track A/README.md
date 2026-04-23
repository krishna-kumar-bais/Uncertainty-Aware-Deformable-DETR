# Track A — Aleatoric Variance Head

Part of the **Uncertainty-Aware Deformable DETR** project
(CS776 course project, IIT Kanpur).

Track A turns the standard Deformable DETR detector into an
**aleatoric-uncertainty-aware detector** by adding a lightweight
variance head on top of the decoder. The head predicts a per-box
`log_var ∈ R^4` (one log-variance per bounding-box coordinate) that is
trained with a numerically stable heteroscedastic regression loss.

The detector weights are **frozen** by default, so Track A is a cheap
fine-tune: only the new `var_embed` MLP is updated.

> For the full multi-track pipeline (Track A + B + C, evaluation,
> calibration, MC-dropout, and uncertainty-weighted fine-tuning) see the
> top-level `README.md` and `Track A+B+C/`. This folder is a
> **self-contained** version of Track A.

## 1. What is in this folder

```text
Track A/
├── README.md                     <- this file
├── LICENSE                       <- Apache 2.0 (inherited from Deformable DETR)
├── Track A.ipynb                 <- Kaggle-ready fine-tuning notebook
├── Track A finetuned.pth         <- Track A checkpoint (variance head trained)
├── benchmark.py                  <- inference-speed benchmark
├── main.py                       <- training / evaluation entry point
├── engine.py                     <- training & evaluation loops
├── requirements.txt              <- Python deps (pycocotools, tqdm, cython, scipy)
├── configs/                      <- shell configs for the base detector
├── datasets/                     <- COCO / panoptic dataset wrappers
├── models/
│   ├── deformable_detr.py        <- DeformableDETR with `var_embed` head
│   ├── deformable_transformer.py
│   ├── backbone.py, matcher.py, position_encoding.py, segmentation.py
│   ├── r50_deformable_detr-checkpoint.pth   <- base pretrained checkpoint
│   └── ops/                      <- custom CUDA deformable-attention op
├── tools/                        <- distributed launch helpers
├── util/                         <- boxes, misc utilities, plotting
└── figs/                         <- Deformable DETR figures (upstream)
```

## 2. Method (Track A in one page)

Let `b̂_q` be the predicted bounding box for decoder query `q` and let
`log σ_q² = var_embed(h_q)` be the log-variance produced by the new MLP
head from the same decoder feature `h_q`. During training, for every
matched query the loss is:

```
L_A(q) = ||b_q - b̂_q||² / σ_q² + log σ_q²
```

with `log σ_q²` clamped to `[log_var_min, log_var_max]` (default
`[-4, 4]`, i.e. variance ∈ `[0.018, 54.6]`) and the final loss clamped
to `[0, 100]` for numerical stability (see `SetCriterion.loss_track_a`
in `models/deformable_detr.py`). NaN / Inf guards zero-out the term if
anything slips through.

Key implementation choices:

- **Parallel head**, not a modified bbox head — the existing `bbox_embed`
  is untouched so COCO AP is preserved.
- **Detector frozen** — `main.py` sets `param.requires_grad = 'var_embed'
  in name`, so only the variance MLP is trained.
- **L2 error** in the numerator (better gradients than L1).
- **Tight log-variance clamp** to keep an untrained head from exploding.
- Works with every Deformable-DETR variant (single-scale, DC5, +iterative
  refinement, ++two-stage) via the same `--track_a` flag.

## 3. Requirements

- Linux, NVIDIA GPU, CUDA ≥ 9.2, GCC ≥ 5.4
- Python ≥ 3.7 (tested on 3.10 / 3.12)
- PyTorch ≥ 1.5.1, torchvision ≥ 0.6.1

```bash
conda create -n deformable_detr python=3.10 pip
conda activate deformable_detr
# Pick the CUDA build matching your driver, e.g.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Compile the deformable-attention CUDA op

```bash
cd ./models/ops
sh ./make.sh
python test.py        # expect "all checking is True"
cd ../..
```

On GPUs without the op available (e.g. Kaggle P100) the notebook falls
back to a pure-PyTorch implementation — correct but slower.

## 4. Dataset — COCO 2017

Download from <https://cocodataset.org> and arrange as:

```text
Track A/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
```

You can also pass `--coco_path /absolute/path/to/coco`.

## 5. Pretrained base checkpoint

Track A fine-tunes **on top of** the official Deformable-DETR R50
checkpoint. A copy is already included at:

```text
models/r50_deformable_detr-checkpoint.pth
```

If you want a fresh copy you can re-download the official one from the
project's Main Results table (see §9) and drop it in the same location.

## 6. Train Track A (variance head only)

Single GPU:

```bash
python main.py \
  --dataset_file coco \
  --coco_path ./data/coco \
  --resume ./models/r50_deformable_detr-checkpoint.pth \
  --output_dir ./exps/track_a \
  --track_a \
  --track_a_loss_coef 0.01 \
  --log_var_min -4.0 --log_var_max 4.0 \
  --lr 2e-6 \
  --batch_size 2 \
  --epochs 10 \
  --clip_max_norm 0.01
```

What `--track_a` does:

| Flag                 | Default | Meaning                                               |
| -------------------- | ------- | ----------------------------------------------------- |
| `--track_a`          | off     | Adds the `var_embed` head, enables `loss_track_a`.    |
| `--track_a_loss_coef`| `1.0`   | Weight of the heteroscedastic loss in the total loss. |
| `--log_var_min`      | `-8.0`  | Lower clamp on predicted `log σ²`.                    |
| `--log_var_max`      | `8.0`   | Upper clamp on predicted `log σ²`.                    |

Recommended starting values (used in the included
`Track A finetuned.pth`): `--track_a_loss_coef 0.01`,
`--log_var_min -4.0`, `--log_var_max 4.0`, `--lr 2e-6`,
`--clip_max_norm 0.01`. The small LR and loss coefficient prevent the
untrained head from destabilising the frozen detector.

When `--track_a` is enabled, `main.py` automatically freezes every
parameter whose name does not contain `var_embed` — you can confirm this
in the console: `number of params: <small number>`.

### Multi-GPU

8 GPUs on one node:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh \
  --resume ./models/r50_deformable_detr-checkpoint.pth \
  --track_a --track_a_loss_coef 0.01 \
  --log_var_min -4.0 --log_var_max 4.0 \
  --output_dir ./exps/track_a
```

Slurm:

```bash
GPUS_PER_NODE=8 ./tools/run_dist_slurm.sh <partition> track_a 8 \
  configs/r50_deformable_detr.sh \
  --resume ./models/r50_deformable_detr-checkpoint.pth \
  --track_a --track_a_loss_coef 0.01 \
  --log_var_min -4.0 --log_var_max 4.0 \
  --output_dir ./exps/track_a
```

### Kaggle notebook

Open `Track A.ipynb` in a Kaggle GPU notebook with the
`awsaf49/coco-2017-dataset` dataset attached. The notebook clones the
repo, installs deps, optionally builds the CUDA op, links COCO under
`./data/coco`, and runs the exact `main.py` command above.

## 7. Evaluate Track A

```bash
python main.py \
  --eval \
  --dataset_file coco \
  --coco_path ./data/coco \
  --resume "./Track A finetuned.pth" \
  --output_dir ./exps/track_a_eval \
  --track_a
```

Standard COCO AP is reported by `pycocotools`. The predicted
`pred_log_vars` (shape `[N, Q, 4]`) are available on `outputs` for any
downstream calibration work; the richer uncertainty-aware evaluation
(ECE, PR-AUC, reliability diagrams, aleatoric/epistemic scores) lives
in `Track A+B+C/Deformable-DETR/`.

## 8. Benchmark inference speed

```bash
python benchmark.py \
  --resume "./Track A finetuned.pth" \
  --batch_size 1 --num_iters 300 --warm_iters 5 \
  --dataset_file coco --coco_path ./data/coco \
  --track_a
```

Prints FPS; Track A adds only an MLP per query so the overhead over the
baseline detector is negligible.

## 9. Base Deformable DETR results (for reference)

Track A starts from the R50 Deformable DETR checkpoint, whose official
numbers on COCO val2017 are:

| Method                                             | Epochs | AP   | AP_S | AP_M | AP_L | Params (M) | FPS  |
| -------------------------------------------------- | :----: | :--: | :--: | :--: | :--: | :--------: | :--: |
| Deformable DETR (single scale)                     |   50   | 39.4 | 20.6 | 43.0 | 55.5 | 34         | 27.0 |
| Deformable DETR (single scale, DC5)                |   50   | 41.5 | 24.1 | 45.3 | 56.0 | 34         | 22.1 |
| **Deformable DETR (R50, multi-scale)**             |   50   | 44.5 | 27.1 | 47.6 | 59.6 | 40         | 15.0 |
| + iterative bounding-box refinement                |   50   | 46.2 | 28.3 | 49.2 | 61.5 | 41         | 15.0 |
| ++ two-stage Deformable DETR                       |   50   | 46.9 | 29.6 | 50.1 | 61.6 | 41         | 14.5 |

Track A preserves these numbers while adding calibrated per-box
aleatoric uncertainty.

