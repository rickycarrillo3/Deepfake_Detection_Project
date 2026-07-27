#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "FreqNet-DeepfakeDetection" ]; then
  git clone https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection.git FreqNet-DeepfakeDetection
fi

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r FreqNet-DeepfakeDetection/requirements.txt
pip install -r requirements.txt
pip install torch torchvision
echo
echo "The pretrained checkpoint (4-classes-freqnet-v2.pth) ships inside the cloned repo already —"
echo "no separate download needed."
