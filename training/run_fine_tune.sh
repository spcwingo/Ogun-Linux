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
echo "[ADVISORY] Training requires >= 11 GB free VRAM on NVIDIA RTX 5070 (12 GB)."
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
STAMP=".venv-train/.pip_train_stamp"
REQ="training/requirements-train.txt"
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
