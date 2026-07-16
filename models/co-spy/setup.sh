#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "CO-SPY" ]; then
  git clone https://github.com/Megum1/CO-SPY.git CO-SPY
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
echo "Downloading pretrained checkpoints..."

cd CO-SPY/pretrained

if [ -f "download.sh" ]; then
    bash download.sh
else
    echo "Error: pretrained/download.sh not found."
    exit 1
fi

echo
echo "Verifying downloaded checkpoints..."

find -name "*_weights.pth"
pip install torch torchvision

echo
echo "Setup complete."

echo "Test from the repo root with:"
echo "	models/co-spy/venv/bin/python run.py --model co-spy --dataset <dataset> --model-arg train_dataset=<train_dataset(sd-v1_4 or progan)>"
