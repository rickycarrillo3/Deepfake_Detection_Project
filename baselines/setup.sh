#!/usr/bin/env bash
# Sets up the shared environment for the TrueFake baselines, per baselines/README.md.
# Run from inside baselines/:
#   cd baselines && bash setup.sh
#
# Covers NPR, MISLNet, R50_nodown, R50_TF and CLIP-D, which share one venv
# because none of them pin conflicting versions. P2G has its own setup.sh —
# it pins torch==2.1.0.
#
# Checkpoints (best.pt in each subdirectory) are NOT downloaded here: they were
# provided directly by the TrueFake authors and are not publicly hosted.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repos we import architectures from.
#    grip-unina's repo serves two baselines: networks/resnet_mod.py provides
#    res50nodown for R50_nodown, and networks/openclipnet.py provides the CLIP
#    linear-probe for CLIP-D.
if [ -d "ClipBased-SyntheticImageDetection" ]; then
    echo "ClipBased-SyntheticImageDetection/ already exists, skipping clone."
else
    git clone https://github.com/grip-unina/ClipBased-SyntheticImageDetection.git \
        ClipBased-SyntheticImageDetection
fi

if [ -d "NPR/NPR-DeepfakeDetection" ]; then
    echo "NPR/NPR-DeepfakeDetection/ already exists, skipping clone."
else
    git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git \
        NPR/NPR-DeepfakeDetection
fi

# 2. Isolated venv. Interpreter used to build it is overridable — on clusters
# where the system python3 is old, load a module and point PYTHON at it, e.g.:
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

# 3. Verify each checkpoint loads into the architecture we claim it has. This
#    is the real acceptance test for this directory: a wrong transcription
#    still produces plausible-looking scores, so it must fail loudly here.
echo ""
echo "Verifying checkpoints..."
venv/bin/python verify_checkpoints.py || {
    echo "Checkpoint verification FAILED — do not trust results from this setup."
    exit 1
}

echo ""
echo "Setup complete. Run with (from repo root):"
echo "  baselines/venv/bin/python run.py --model tf-npr    --dataset <dataset_name>"
echo "  baselines/venv/bin/python run.py --model tf-r50e2p --dataset <dataset_name>"
echo "See 'run.py --list' for all registered baselines."
