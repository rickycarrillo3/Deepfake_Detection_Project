#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SETUP_DIR="$(pwd)"

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv

# Conda/miniforge-distributed Python builds sometimes omit the ensurepip
# bundled wheel files, so `python -m venv` reports success but silently
# produces a venv with no working venv/bin/pip. Same guard used by every
# other model's setup.sh in this repo.
if [ ! -x "$SETUP_DIR/venv/bin/pip" ] || ! "$SETUP_DIR/venv/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "venv/bin/pip missing or broken after venv creation (known conda/miniforge issue) — bootstrapping pip manually..."
  "$SETUP_DIR/venv/bin/python" -m ensurepip --upgrade --default-pip 2>/dev/null || {
    echo "ensurepip unavailable in this Python build — falling back to get-pip.py"
    curl -sS https://bootstrap.pypa.io/get-pip.py -o "/tmp/get-pip-$$.py"
    "$SETUP_DIR/venv/bin/python" "/tmp/get-pip-$$.py"
    rm -f "/tmp/get-pip-$$.py"
  }
fi
if [ ! -x "$SETUP_DIR/venv/bin/pip" ]; then
  echo "ERROR: could not bootstrap a working pip into venv/. Aborting." >&2
  exit 1
fi
PIP="$SETUP_DIR/venv/bin/pip"

"$PIP" install --upgrade pip -q
"$PIP" install scikit-learn pandas joblib numpy

echo
echo "No pretrained weights to download -- this model is trained from scratch on your"
echo "own already-computed results/*.csv scores (no GPU, no image pixels needed for"
echo "training -- just each expert's score). Pick 3-5 expert models and train with:"
echo "  models/rf_ensemble/venv/bin/python train.py \\"
echo "    --datasets <dataset...> --stems <expert_stem...> --out models/rf_ensemble/rf_ensemble.joblib"
echo
echo "Then run it exactly like every other model:"
echo "  models/rf_ensemble/venv/bin/python run.py --model rf_ensemble --dataset <dataset>"
