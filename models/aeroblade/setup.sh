#!/usr/bin/env bash
# Sets up the isolated environment for models/aeroblade/, per README.md.
# Run from inside models/aeroblade/:
#   cd models/aeroblade && bash setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (contains src/aeroblade/, scripts/, setup.py)
if [ -d "aeroblade" ]; then
    echo "aeroblade/ already exists, skipping clone."
else
    git clone https://github.com/jonasricker/aeroblade.git aeroblade
fi

# 2. Isolated venv, Python 3.10 (matches upstream's tested version)
if [ -d "aeroblade_env" ]; then
    echo "aeroblade_env/ already exists, skipping venv creation."
elif command -v python3.10 >/dev/null 2>&1; then
    python3.10 -m venv aeroblade_env
elif command -v conda >/dev/null 2>&1; then
    echo "python3.10 not found on PATH; creating the env via conda instead."
    conda create --prefix ./aeroblade_env python=3.10 pip -y
else
    echo "ERROR: python3.10 not found on PATH, and conda is not available to install it."
    echo "Install Python 3.10 first, e.g.:"
    echo "  macOS (Homebrew): brew install python@3.10"
    echo "  pyenv:            pyenv install 3.10.16 && pyenv local 3.10.16"
    echo "then re-run this script."
    exit 1
fi

aeroblade_env/bin/pip install --upgrade pip

if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing PyTorch separately and filtering CUDA-only packages."

    # Install macOS PyTorch (CPU/MPS)
    aeroblade_env/bin/pip install torch==2.1.2 torchvision==0.16.2

    # Remove packages that are Linux/CUDA-only or already installed above.
    grep -vE '^(torch==|torchvision==|triton==|nvidia-)' \
        aeroblade/requirements.txt > /tmp/aeroblade_requirements.txt

    aeroblade_env/bin/pip install -r /tmp/aeroblade_requirements.txt
else
    aeroblade_env/bin/pip install -r aeroblade/requirements.txt
fi

aeroblade_env/bin/pip install -e ./aeroblade
aeroblade_env/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. First run will download AE weights from HuggingFace"
echo "(CompVis/stable-diffusion-v1-1, stabilityai/stable-diffusion-2-base,"
echo "kandinsky-community/kandinsky-2-1) — make sure you're logged in via"
echo "'aeroblade_env/bin/huggingface-cli login' if any require accepting a license."
echo ""
echo "Run with:"
echo "  PYTHONPATH=<repo_root> aeroblade_env/bin/python run.py <dataset_name>"
