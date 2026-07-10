#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "AIDE" ]; then
  git clone https://github.com/shilinyan99/AIDE.git AIDE
fi

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo
echo "Torch/torchvision are not installed by this script — install them separately, matched to"
echo "your local CUDA version, e.g.:"
echo "  source models/aide/venv/bin/activate && pip install torch torchvision"
echo
echo "AIDE's own requirements may pin older torch versions, so this script only installs"
echo "models/aide/requirements.txt. Do not blindly pip install AIDE/requirements.txt unless"
echo "you intentionally want to change your torch/CUDA setup."
echo
echo "Then download or place an AIDE pretrained detector checkpoint, e.g.:"
echo "  mkdir -p models/aide/AIDE/checkpoints"
echo "  # put GenImage_train.pth at:"
echo "  # models/aide/AIDE/checkpoints/GenImage_train.pth"
echo
echo "After that, test from the repo root with:"
echo "  source models/aide/venv/bin/activate"
echo "  python models/aide/run.py --dataset flux_faces --force-label 1 --limit 10"
echo
echo "For the full fake-only run:"
echo "  python models/aide/run.py --dataset flux_faces --force-label 1"