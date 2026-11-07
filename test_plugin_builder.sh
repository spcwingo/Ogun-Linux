#!/usr/bin/env bash

# Resolve and cd to the directory containing this script (repo root)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

VENV_PY=".venv/bin/python"

# 1. Virtual Environment Bootstrap
if [ ! -x "$VENV_PY" ]; then
    echo "[*] Creating virtual environment .venv..."
    if ! python3.12 -m venv .venv 2>/dev/null; then
        python3 -m venv .venv
    fi
fi

# 2. Pip Requirement Timestamp Cache
STAMP=".venv/.pip_stamp"
REQ="requirements.txt"
NEED_INSTALL=0

if [ ! -f "$STAMP" ]; then
    NEED_INSTALL=1
elif [ "$REQ" -nt "$STAMP" ]; then
    # reinstall only when the requirements file is NEWER than the stamp
    # written after the last install
    NEED_INSTALL=1
fi

if [ "$NEED_INSTALL" -eq 1 ]; then
    echo "[*] Installing dependencies from $REQ..."
    "$VENV_PY" -m pip install -r "$REQ"
    PIP_EXIT=$?
    if [ $PIP_EXIT -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit $PIP_EXIT
    fi
    touch "$STAMP"
fi

# 3. Execution Environment
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export PYTHONPATH="$SCRIPT_DIR"

"$VENV_PY" test_harness.py "$@"
EXITCODE=$?

exit $EXITCODE
