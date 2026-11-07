#!/usr/bin/env bash

# Resolve the repo root (parent of this script's directory) and cd to it
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR" || exit 1

# Determine Python interpreter: use root .venv if present, otherwise system python3.12/python3
if [ -x ".venv/bin/python" ]; then
    PY_EXE=".venv/bin/python"
elif command -v python3.12 >/dev/null 2>&1; then
    PY_EXE="python3.12"
else
    PY_EXE="python3"
fi

echo "[*] Running Ogun DPO Dataset Preparation Pipeline..."
"$PY_EXE" training/prepare_training_data.py "$@"
EXITCODE=$?

if [ "$EXITCODE" -eq 0 ]; then
    echo
    echo "[+] Dataset preparation complete."
    echo "[+] Merged dataset: training/data/merged_dpo_dataset.jsonl"
    echo "[+] Statistics:     training/data/merged_dpo_dataset.stats.json"
else
    echo
    echo "[!] Dataset preparation exited with error code $EXITCODE."
fi

exit $EXITCODE
