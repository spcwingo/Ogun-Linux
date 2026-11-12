#!/usr/bin/env bash

# Capture the directory Ogun was launched from, before cd-ing to the repo
# root: when Ogun is started inside another project, the file tools anchor
# to that project ("project mode"). An OGUN_WORK_DIR already set in the
# environment wins over the launch directory.
if [ -z "${OGUN_WORK_DIR:-}" ]; then
    OGUN_WORK_DIR="$PWD"
fi
export OGUN_WORK_DIR

# Resolve and cd to the directory containing this script (repo root)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

VENV_PY=".venv/bin/python"

# 1. Virtual Environment Bootstrap
# A venv is only as good as the Python that created it. A stale venv whose
# base Python moved, or a Python whose own venvs come out broken (some
# standalone/uv-managed builds resolve the new interpreter's standard
# library to a leftover build-machine path), yields a python that dies
# with "No module named 'encodings'". So: reuse .venv only if its
# interpreter actually runs; otherwise rebuild, trying each Python
# candidate in turn and verifying the result before accepting it.
venv_healthy() {
    [ -x "$VENV_PY" ] && "$VENV_PY" -c "import encodings" >/dev/null 2>&1
}

try_create_venv() {
    # $1 = python executable; returns 0 only if it leaves a WORKING venv
    rm -rf .venv
    echo "[*] Creating virtual environment .venv with $1..."
    "$1" -m venv .venv >/dev/null 2>&1 || return 1
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
        echo "[*] Creating virtual environment .venv with uv..."
        rm -rf .venv
        if uv venv .venv --python 3.12 --seed >/dev/null 2>&1 && venv_healthy; then
            CREATED=1
        fi
    fi
    if [ "$CREATED" -eq 0 ]; then
        echo "[ERROR] No available Python could create a working virtual environment." >&2
        echo "        The Python installation itself is producing broken venvs." >&2
        echo "        Fixes: run 'uv venv .venv --python 3.12 --seed' by hand, then re-run this script;" >&2
        echo "        or install your distribution's python3.12 and python3.12-venv packages." >&2
        rm -rf .venv
        exit 1
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

"$VENV_PY" main.py "$@"
EXITCODE=$?

exit $EXITCODE
