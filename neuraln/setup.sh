#!/usr/bin/env bash
# Sets up the isolated environment for the neural-network ensemble, per README.md.
# Run from inside neural_network/:
#   cd neural_network && bash setup.sh
#
# Optional: every model venv in models/ already has torch + pandas, so you can
# skip this and run the trainer under one of those:
#   models/freqnet/venv/bin/python neural_network/train.py ...
# This venv just avoids depending on which detector happens to be installed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Interpreter used to build the venv is overridable — on clusters where the
# system python3 is old, load a module and point PYTHON at it, e.g.:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip

# CPU-only torch: the network is a few hundred parameters over 3-5 features and
# trains in seconds on a laptop. No CUDA wheel index, no torchvision.
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing plain (CPU/MPS) torch."
    venv/bin/pip install torch
else
    venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. From the repo root:"
echo "  neural_network/venv/bin/python neural_network/train.py --list"
echo "  neural_network/venv/bin/python neural_network/train.py \\"
echo "      --models <stem1>,<stem2>,<stem3> --dry-run"
