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

# 2. Isolated venv. Interpreter used to build it is overridable — upstream
# tests on Python 3.10, so on clusters load a matching module and point PYTHON
# at it, e.g.:
#   module load python/3.10 && PYTHON=python3.10 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "aeroblade_env" ]; then
    echo "aeroblade_env/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv aeroblade_env
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
