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

# 2. Isolated venv, Python 3.9 (matches upstream's pinned torch build)
if [ -d "rine_env" ]; then
    echo "rine_env/ already exists, skipping venv creation."
else
    python3.9 -m venv rine_env
fi

rine_env/bin/pip install --upgrade pip
rine_env/bin/pip install torch==2.1.1 torchvision==0.16.1 \
    --index-url https://download.pytorch.org/whl/cu118
rine_env/bin/pip install -r rine/requirements.txt
rine_env/bin/pip install -r requirements.txt

# 3. numpy gets bumped to 2.x by one of the above installs; RINE's
#    src/utils needs 1.26.4, so pin it back down last.
rine_env/bin/pip uninstall -y numpy
rine_env/bin/pip install numpy==1.26.4

echo ""
echo "Setup complete. Run with:"
echo "  PYTHONPATH=<repo_root> rine_env/bin/python run.py <dataset_name>"
