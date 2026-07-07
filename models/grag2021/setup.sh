#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "GANimageDetection" ]; then
  git clone https://github.com/grip-unina/GANimageDetection.git GANimageDetection
fi

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

mkdir -p GANimageDetection/weights
WEIGHTS_PATH="GANimageDetection/weights/gandetection_resnet50nodown_stylegan2.pth"
if [ ! -f "$WEIGHTS_PATH" ]; then
  wget -O "$WEIGHTS_PATH" \
    https://www.grip.unina.it/download/prog/GANdetection/weights/gandetection_resnet50nodown_stylegan2.pth
fi

echo
echo "Torch/torchvision are not installed by this script — install them separately, matched to"
echo "your local CUDA version, e.g.:"
echo "  source models/grag2021/venv/bin/activate && pip install torch torchvision"
