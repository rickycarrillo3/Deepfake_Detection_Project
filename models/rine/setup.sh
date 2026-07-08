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

# 2. Isolated venv, Python 3.9 (matches upstream's pinned torch build)
if [ -d "rine_env" ]; then
    echo "rine_env/ already exists, skipping venv creation."
elif command -v python3.9 >/dev/null 2>&1; then
    python3.9 -m venv rine_env
elif command -v conda >/dev/null 2>&1; then
    echo "python3.9 not found on PATH; creating the env via conda instead."
    conda create --prefix ./rine_env python=3.9 pip -y
else
    echo "ERROR: python3.9 not found on PATH, and conda is not available to install it."
    echo "Install Python 3.9 first, e.g.:"
    echo "  macOS (Homebrew): brew install python@3.9"
    echo "  pyenv:            pyenv install 3.9.18 && pyenv local 3.9.18"
    echo "then re-run this script."
    exit 1
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
echo "  PYTHONPATH=<repo_root> rine_env/bin/python run.py <dataset_name>"
