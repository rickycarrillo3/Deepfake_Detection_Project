#!/usr/bin/env bash
# Sets up the isolated environment for the P2G baseline, per baselines/README.md.
# Run from inside baselines/P2G/:
#   cd baselines/P2G && bash setup.sh
#
# P2G gets its own venv rather than sharing baselines/venv because it pins
# torch==2.1.0 (and torchvision 0.16.0) — exactly the kind of conflicting pin
# the per-model isolation in models/ exists for.
#
# The checkpoint (best.pt) is NOT downloaded here: it was provided directly by
# the TrueFake authors and is not publicly hosted. The frozen CLIP ViT-B/16
# backbone downloads itself on first run.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (src/models/slinet.py, its CLIP fork, and the
#    ImageNet-1k class names used for the zero-shot conditioning pass).
if [ -d "Prompt2Guard" ]; then
    echo "Prompt2Guard/ already exists, skipping clone."
else
    git clone https://github.com/laitifranz/Prompt2Guard.git Prompt2Guard
fi

# 2. Isolated venv. Interpreter is overridable for cluster module systems:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "Verifying checkpoint..."
venv/bin/python ../verify_checkpoints.py P2G || {
    echo "Checkpoint verification FAILED — do not trust results from this setup."
    exit 1
}

echo ""
echo "Setup complete. Run with (from repo root):"
echo "  baselines/P2G/venv/bin/python run.py --model tf-p2g --dataset <dataset_name>"
