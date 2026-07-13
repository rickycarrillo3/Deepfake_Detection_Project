#!/usr/bin/env bash
# Sets up the isolated environment for models/cnndetection/, per README.md.
# Run from inside models/cnndetection/:
#   cd models/cnndetection && bash setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (networks/resnet.py, demo.py, weights/download_weights.sh)
if [ -d "CNNDetection" ]; then
    echo "CNNDetection/ already exists, skipping clone."
else
    git clone https://github.com/PeterWang512/CNNDetection.git CNNDetection
fi

# 2. Isolated venv. Interpreter used to build it is overridable — on clusters
# where the system python3 is old, load a module and point PYTHON at it, e.g.:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "cnndetection_env" ]; then
    echo "cnndetection_env/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv cnndetection_env
fi

cnndetection_env/bin/pip install --upgrade pip
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing plain (CPU/MPS) torch, no CUDA wheel index."
    cnndetection_env/bin/pip install torch torchvision
else
    cnndetection_env/bin/pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118
fi
cnndetection_env/bin/pip install -r requirements.txt

# 3. Weights — download_weights.sh (shipped in the repo) fetches both
#    blur_jpg_prob0.5.pth and blur_jpg_prob0.1.pth into CNNDetection/weights/.
#    model.py defaults to the 0.5 variant.
if [ -f "CNNDetection/weights/blur_jpg_prob0.5.pth" ]; then
    echo "Weights already present, skipping download."
else
    echo "Downloading weights..."
    ( cd CNNDetection && bash weights/download_weights.sh )
fi

echo ""
echo "Setup complete. Run with:"
echo "  (from repo root) models/cnndetection/cnndetection_env/bin/python run.py --model cnndetection --dataset <dataset_name>"
echo "Single image:"
echo "  cnndetection_env/bin/python predict_one.py /path/to/image.png"
