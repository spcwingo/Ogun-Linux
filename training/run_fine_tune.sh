#!/usr/bin/env bash

# Resolve the repo root (parent of this script's directory) and cd to it
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR" || exit 1

VENV_PY=".venv-train/bin/python"

# 1. Check for prerequisite merged dataset file
DATASET_FILE="training/data/merged_dpo_dataset.jsonl"
if [ ! -f "$DATASET_FILE" ]; then
    echo "[ERROR] Required dataset file '$DATASET_FILE' is missing."
    echo "Please run 'training/run_data_prep.sh' first to download and prepare training data."
    exit 1
fi

# 2. Hardware advisory
echo "======================================================================"
echo " Ogun Unsloth DPO Fine-Tuning Launcher"
echo "======================================================================"
echo "[ADVISORY] Training requires >= 11 GB free VRAM on a 12 GB-class GPU (NVIDIA CUDA or AMD ROCm)."
echo "Please close Ollama, Blender, Unreal Engine, and browser hardware acceleration"
echo "to prevent GPU out-of-memory (OOM) errors."
echo "======================================================================"

# 3. Virtual environment setup for training (.venv-train)
if [ ! -x "$VENV_PY" ]; then
    echo "[*] Creating dedicated training virtual environment .venv-train..."
    if ! python3.12 -m venv .venv-train 2>/dev/null; then
        python3 -m venv .venv-train
    fi
fi

# 4. Dependency caching
# Select the dependency set for the GPU backend: ROCm when requested
# (OGUN_TRAIN_BACKEND=rocm) or when rocm-smi is present without nvidia-smi.
case "${OGUN_TRAIN_BACKEND:-auto}" in
    rocm) TRAIN_BACKEND="rocm" ;;
    cuda) TRAIN_BACKEND="cuda" ;;
    *)
        if command -v rocm-smi >/dev/null 2>&1 && ! command -v nvidia-smi >/dev/null 2>&1; then
            TRAIN_BACKEND="rocm"
        else
            TRAIN_BACKEND="cuda"
        fi
        ;;
esac
if [ "$TRAIN_BACKEND" = "rocm" ]; then
    REQ="training/requirements-train-rocm.txt"
else
    REQ="training/requirements-train.txt"
fi
echo "[*] Training backend: $TRAIN_BACKEND (dependencies: $REQ)"

STAMP=".venv-train/.pip_train_stamp"
NEED_INSTALL=0

if [ ! -f "$STAMP" ]; then
    NEED_INSTALL=1
elif [ "$REQ" -nt "$STAMP" ]; then
    # reinstall only when the requirements file is NEWER than the stamp
    # written after the last install
    NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[*] Installing training dependencies from $REQ..."
    "$VENV_PY" -m pip install -r "$REQ"
    PIP_EXIT=$?
    if [ $PIP_EXIT -ne 0 ]; then
        echo "[ERROR] Failed to install training dependencies."
        exit $PIP_EXIT
    fi
    touch "$STAMP"
fi

# 5. Run trainer
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$ROOT_DIR:$SCRIPT_DIR"

"$VENV_PY" training/ogun_unsloth_trainer.py "$@"
EXITCODE=$?

exit $EXITCODE
