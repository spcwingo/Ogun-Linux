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
# Same rule as run_ogun.sh: reuse the venv only if its interpreter
# actually runs; otherwise rebuild from the first Python candidate that
# produces a WORKING venv (verifying with a stdlib import each time).
venv_healthy() {
    [ -x "$VENV_PY" ] && "$VENV_PY" -c "import encodings" >/dev/null 2>&1
}

try_create_venv() {
    # $1 = python executable; returns 0 only if it leaves a WORKING venv
    rm -rf .venv-train
    echo "[*] Creating dedicated training virtual environment .venv-train with $1..."
    "$1" -m venv .venv-train >/dev/null 2>&1 || return 1
    venv_healthy
}

if ! venv_healthy; then
    CREATED=0
    for CAND in python3.12 python3 /usr/bin/python3.12 /usr/bin/python3; do
        if [ -x "$CAND" ] || command -v "$CAND" >/dev/null 2>&1; then
            if try_create_venv "$CAND"; then
                CREATED=1
                break
            fi
        fi
    done
    # Last resort: uv (whose managed Pythons are one source of broken
    # venvs) knows how to build a working venv from them.
    if [ "$CREATED" -eq 0 ] && command -v uv >/dev/null 2>&1; then
        echo "[*] Creating dedicated training virtual environment .venv-train with uv..."
        rm -rf .venv-train
        if uv venv .venv-train --python 3.12 --seed >/dev/null 2>&1 && venv_healthy; then
            CREATED=1
        fi
    fi
    if [ "$CREATED" -eq 0 ]; then
        echo "[ERROR] No available Python could create a working virtual environment." >&2
        echo "        The Python installation itself is producing broken venvs." >&2
        echo "        Fixes: run 'uv venv .venv-train --python 3.12 --seed' by hand, then re-run this script;" >&2
        echo "        or install your distribution's python3.12 and python3.12-venv packages." >&2
        rm -rf .venv-train
        exit 1
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
