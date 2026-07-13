#!/usr/bin/env bash
# Sets up the isolated environment for models/rine/, per README.md.
# Run from inside models/rine/:
#   cd models/rine && bash setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (contains src/, ckpt/, demo/, scripts/)
if [ -d "rine" ]; then
    echo "rine/ already exists, skipping clone."
else
    git clone https://github.com/mever-team/rine.git rine
fi

# The upstream code expects checkpoints at ./ckpt/, but the repository
# stores them in ./rine/ckpt/. Create a symlink if needed.
if [ ! -e "ckpt" ] && [ -d "rine/ckpt" ]; then
    echo "Creating ckpt -> rine/ckpt symlink..."
    ln -s rine/ckpt ckpt
fi

# 2. Isolated venv. Interpreter used to build it is overridable — upstream
# pins its torch build against Python 3.9, so on clusters load a matching
# module and point PYTHON at it, e.g.:
#   module load python/3.9 && PYTHON=python3.9 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "rine_env" ]; then
    echo "rine_env/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv rine_env
fi

rine_env/bin/pip install --upgrade pip
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing plain (CPU/MPS) torch, no CUDA wheel index."
    rine_env/bin/pip install torch==2.1.1 torchvision==0.16.1
else
    rine_env/bin/pip install torch==2.1.1 torchvision==0.16.1 \
        --index-url https://download.pytorch.org/whl/cu118
fi

rine_env/bin/pip install -r rine/requirements.txt
rine_env/bin/pip install -r requirements.txt

# 3. numpy gets bumped to 2.x by one of the above installs; RINE's
#    src/utils needs 1.26.4, so pin it back down last.
rine_env/bin/pip uninstall -y numpy
rine_env/bin/pip install numpy==1.26.4

echo ""
echo "Setup complete. Run with:"
echo "  (from repo root) models/rine/rine_env/bin/python run.py --model rine --dataset <dataset_name>"
