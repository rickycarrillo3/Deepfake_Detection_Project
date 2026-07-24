#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "TrueFake-IJCNN25" ]; then
  git clone https://github.com/MMLab-unitn/TrueFake-IJCNN25.git TrueFake-IJCNN25
fi

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
# Upstream ships a conda environment.yml (linux/CUDA-pinned). requirements.txt
# here is the pip-installable equivalent of the packages its code actually
# imports, so the venv builds on macOS and on the cluster alike.
pip install -r requirements.txt

# The pretrained R50-E2P checkpoint is a Google Drive download (see the upstream
# README's "Use pretrained network"). gdown handles Drive's large-file confirm
# interstitial, which a plain curl does not.
mkdir -p weights
if [ ! -f "weights/best.pt" ]; then
  # Bare id, not `--id` — gdown removed that flag in 6.x.
  gdown 1m0b8HZuOCcF_-l2-GJBi3C22RuG5UDvU -O weights/best.pt
else
  echo "weights/best.pt already present — skipping download."
fi

echo
echo "Done. Run with:"
echo "  baselines/r50_e2p/venv/bin/python run.py --model r50_e2p --dataset ffhq_test"
