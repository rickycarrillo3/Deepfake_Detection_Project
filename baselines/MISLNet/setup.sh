#!/usr/bin/env bash
# Sets up the isolated environment for the tf-mislnet baseline, per baselines/README.md.
# Run from inside baselines/MISLNet/:
#   cd baselines/MISLNet && bash setup.sh
#
# Each baseline keeps its own venv, matching the per-directory isolation used
# throughout models/. The checkpoint (best.pt) is NOT downloaded here: it was
# provided directly by the TrueFake authors and is not publicly hosted.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Isolated venv. Interpreter is overridable for cluster module systems:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing plain (CPU/MPS) torch, no CUDA wheel index."
    venv/bin/pip install torch torchvision
else
    venv/bin/pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118
fi
venv/bin/pip install -r requirements.txt

# Verify the checkpoint loads into the architecture we claim it has. This is
# the acceptance test: a wrong transcription still produces plausible-looking
# scores, so it must fail loudly here.
echo ""
echo "Verifying checkpoint..."
venv/bin/python ../verify_checkpoints.py MISLNet || {
    echo "Checkpoint verification FAILED — do not trust results from this setup."
    exit 1
}

echo ""
echo "Setup complete. Run with (from repo root):"
echo "  baselines/MISLNet/venv/bin/python run.py --model tf-mislnet --dataset <dataset_name>"
