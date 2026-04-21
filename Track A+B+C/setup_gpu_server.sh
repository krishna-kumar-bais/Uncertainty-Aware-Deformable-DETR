#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$PWD}"
CUDA_FLAVOR="${CUDA_FLAVOR:-cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

cd "$PROJECT_ROOT"

echo "[1/8] Checking Python version"
$PYTHON_BIN - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) not in {(3, 10), (3, 11)}:
    raise SystemExit(f"Expected Python 3.10 or 3.11, found {major}.{minor}")
print(f"Python version OK: {major}.{minor}")
PY

echo "[2/8] Creating virtual environment at $VENV_DIR"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/8] Installing pinned packaging tools"
python -m pip install --upgrade "pip==24.2" "setuptools==75.1.0" "wheel==0.44.0"

echo "[4/8] Installing exact PyTorch build for $CUDA_FLAVOR"
python -m pip install \
  "torch==2.4.1" \
  "torchvision==0.19.1" \
  "torchaudio==2.4.1" \
  --index-url "https://download.pytorch.org/whl/$CUDA_FLAVOR"

echo "[5/8] Installing exact project dependencies"
python -m pip install -r Deformable-DETR/requirements.txt

echo "[6/8] Building Deformable DETR CUDA operators"
pushd Deformable-DETR/models/ops >/dev/null
python setup.py build install
popd >/dev/null

echo "[7/8] Running operator unit test"
pushd Deformable-DETR/models/ops >/dev/null
python test.py
popd >/dev/null

echo "[8/8] Validating exact environment"
python validate_gpu_env.py --expect-cuda

echo
echo "GPU server setup completed successfully."
echo "Activate later with: source \"$VENV_DIR/bin/activate\""
