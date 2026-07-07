#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "CO-SPY" ]; then
  git clone https://github.com/Megum1/CO-SPY.git CO-SPY
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

echo
echo "Torch/torchvision are not installed by this script — install them separately, matched to"
echo "your local CUDA version, e.g.:"
echo "  source models/co-spy/venv/bin/activate && pip install torch torchvision"
echo
echo "Then download pretrained checkpoints, e.g.:"
echo "  cd models/co-spy/CO-SPY && sh pretrained/download.sh"
echo "so that models/co-spy/CO-SPY/pretrained/<train_dataset>/{semantic,artifact,fusion}_weights.pth exist."
