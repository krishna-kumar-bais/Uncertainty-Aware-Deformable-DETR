# GPU Server Setup

This guide is the safest path to move the project to a GPU Linux server, install the exact dependencies, compile the CUDA operators, and validate that the environment matches the pinned versions.

## 1. Copy the project from local machine to server

From your Windows machine, use `scp`:

```powershell
scp -r "C:\Users\MyPc\Downloads\CS776_Project\CS776_Project" user@server:/path/to/workdir
```

After copying, log into the server:

```bash
ssh user@server
cd /path/to/workdir/CS776_Project
```

## 2. Create and validate the GPU environment

The setup script installs exact versions, builds the Deformable DETR CUDA ops, runs the operator unit test, and validates the environment.

CUDA 12.1 example:

```bash
CUDA_FLAVOR=cu121 bash setup_gpu_server.sh
```

CUDA 11.8 example:

```bash
CUDA_FLAVOR=cu118 bash setup_gpu_server.sh
```

What the script does:

1. Checks that Python is `3.10` or `3.11`.
2. Creates `.venv`.
3. Pins packaging tools:
   - `pip==24.2`
   - `setuptools==75.1.0`
   - `wheel==0.44.0`
4. Installs exact PyTorch GPU wheels.
5. Installs exact project dependencies from [requirements.txt](C:/Users/MyPc/Downloads/CS776_Project/CS776_Project/Deformable-DETR/requirements.txt:1).
6. Builds the custom CUDA operator under `Deformable-DETR/models/ops`.
7. Runs `python test.py` for the operator.
8. Verifies imports and exact package versions with [validate_gpu_env.py](C:/Users/MyPc/Downloads/CS776_Project/CS776_Project/validate_gpu_env.py:1).

## 3. Baseline evaluation

Run the deterministic baseline first. Use `--eval_uncertainty` here too so that the baseline and improved runs are saved in the same format.

```bash
source .venv/bin/activate

python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --uncertainty_score confidence \
  --resume Deformable-DETR/models/r50_deformable_detr-checkpoint.pth \
  --coco_path /path/to/coco \
  --output_dir outputs/base_eval
```

This writes:

- `outputs/base_eval/eval_metrics.json`
- `outputs/base_eval/uncertainty_eval/metrics.json`

## 4. Improved Track B evaluation

Example with `T=10` decoder-FFN MC dropout:

```bash
python Deformable-DETR/main.py \
  --eval \
  --eval_uncertainty \
  --mc_dropout \
  --mc_runs 10 \
  --mc_dropout_scope decoder_ffn \
  --uncertainty_score combined \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path /path/to/coco \
  --output_dir outputs/track_b_t10
```

## 5. Compare baseline vs improved

```bash
python compare_runs.py \
  --base outputs/base_eval \
  --improved outputs/track_b_t10
```

This prints:

- `AP`, `AP50`, `AP75`, `AP_S`, `AP_M`, `AP_L`
- `ECE`
- `Uncertainty_PR_AUC`
- mean confidence / mean accuracy

For `ECE`, the comparison script reports the delta so that a positive value means the improved model reduced calibration error.

## 6. Search for the smallest useful T

Instead of assuming `T=10`, sweep the lower values and keep the smallest `T` that preserves AP while improving calibration:

```bash
python sweep_mc_dropout.py \
  --resume outputs/track_a/checkpoint.pth \
  --coco_path /path/to/coco \
  --output_root outputs/mc_sweep
```

Start with:

- `T = 2, 3, 5, 8, 10`
- temperature = `1.0, 1.25, 1.5`
- uncertainty score = `combined`, `entropy`

That gives you a clean research claim: the improved model achieves better uncertainty quality than the base model at the lowest practical stochastic budget.
