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

# 2. Isolated env at venv/, as every other model has.
#
# This model constrains its interpreter more than the others do: upstream pins
# torch==2.1.2, whose wheels stop at cp311, so Python >= 3.12 cannot install
# this stack at all. The cluster's default python3 is 3.12, so rather than make
# that a documented manual step, this script sources a suitable interpreter
# itself when the default one is too new and conda (miniforge) is available —
# creating venv/ *as* a conda env, so the path, the layout and run.py's venv
# discovery are identical either way.
#
# PYTHON= still overrides everything and is used as-is (only warned about), per
# the repo convention that a cluster's module-loaded interpreter can be chosen:
#   module load python/3.10 && PYTHON=python3.10 ./setup.sh
PYTHON_EXPLICIT="${PYTHON:+yes}"
PYTHON="${PYTHON:-python3}"
CONDA_PY_VERSION="3.10"  # upstream's tested version

# 0 if the interpreter can install torch==2.1.2 (i.e. < 3.12), 1 otherwise.
python_supported() {
    "$1" -c 'import sys; sys.exit(0 if sys.version_info < (3, 12) else 1)' 2>/dev/null
}

# Path to a usable conda/mamba, or non-zero if there is none. $CONDA_EXE first:
# module systems commonly expose conda as a *shell function* that a script like
# this never sees, while still exporting CONDA_EXE.
conda_bin() {
    if [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE}" ]; then
        echo "${CONDA_EXE}"
        return 0
    fi
    command -v conda 2>/dev/null || command -v mamba 2>/dev/null || return 1
}

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping env creation."
elif python_supported "$PYTHON"; then
    echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"
    "$PYTHON" -m venv venv
elif [ -n "$PYTHON_EXPLICIT" ]; then
    # Explicit choice wins, but say plainly what is about to break.
    echo "WARNING: PYTHON=$PYTHON is $("$PYTHON" --version | cut -d' ' -f2), but upstream pins" >&2
    echo "         torch==2.1.2, which has no wheels for Python 3.12+. Continuing as asked;" >&2
    echo "         expect pip to fail. Unset PYTHON to let this script pick an interpreter." >&2
    "$PYTHON" -m venv venv
elif CONDA="$(conda_bin)"; then
    echo "Default $("$PYTHON" --version) is too new for torch==2.1.2 (needs < 3.12)."
    echo "Creating venv/ as a Python $CONDA_PY_VERSION conda env with $CONDA instead."
    "$CONDA" create -y -p venv "python=$CONDA_PY_VERSION" pip
else
    echo "ERROR: $("$PYTHON" --version) is too new — upstream pins torch==2.1.2, which has" >&2
    echo "       no wheels for Python 3.12+, and no conda/mamba was found to create an" >&2
    echo "       older one. Load a module providing conda (e.g. 'module load miniforge')," >&2
    echo "       or point PYTHON at a Python 3.10/3.11 interpreter." >&2
    exit 1
fi

echo "venv/ interpreter: $(venv/bin/python --version)"

venv/bin/pip install --upgrade pip

if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing PyTorch separately and filtering CUDA-only packages."

    # Install macOS PyTorch (CPU/MPS)
    venv/bin/pip install torch==2.1.2 torchvision==0.16.2

    # Remove packages that are Linux/CUDA-only or already installed above.
    grep -vE '^(torch==|torchvision==|triton==|nvidia-)' \
        aeroblade/requirements.txt > /tmp/aeroblade_requirements.txt

    venv/bin/pip install -r /tmp/aeroblade_requirements.txt
else
    venv/bin/pip install -r aeroblade/requirements.txt
fi

venv/bin/pip install -e ./aeroblade
venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. First run will download AE weights from HuggingFace"
echo "(CompVis/stable-diffusion-v1-1, stabilityai/stable-diffusion-2-base,"
echo "kandinsky-community/kandinsky-2-1) — make sure you're logged in via"
echo "'venv/bin/huggingface-cli login' if any require accepting a license."
echo ""
echo "Run with (from the repo root, via the shared runner):"
echo "  models/aeroblade/venv/bin/python run.py --model aeroblade --dataset <dataset_name>"
