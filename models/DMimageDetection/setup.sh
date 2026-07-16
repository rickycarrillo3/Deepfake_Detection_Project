#!/usr/bin/env bash
# Sets up the shared environment for models/DMimageDetection/, used by both
# grag2021_progan/ and grag2021_latent/.
# Run from inside models/DMimageDetection/:
#   cd models/DMimageDetection && bash setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (get_method_here.py, normalization*.py,
#    networks/networks/resnet_mod.py, etc.) — shared by both variants.
if [ -d "dmimagedetection" ]; then
    echo "dmimagedetection/ already exists, skipping clone."
else
    git clone https://github.com/grip-unina/DMimageDetection.git dmimagedetection
fi

# 2. One shared venv for both variants (identical deps). Interpreter used to
# build it is overridable — on clusters where the system python3 is old, load
# a module and point PYTHON at it, e.g.:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip
if [ "$(uname -s)" = "Darwin" ]; then
    echo "macOS detected — installing plain (CPU/MPS) torch, no CUDA wheel index."
    venv/bin/pip install torch torchvision
else
    venv/bin/pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118
fi
venv/bin/pip install -r requirements.txt

# 3. Weights — both variants ship in the same zip, downloaded once here.
#    Expected layout after extraction:
#      weights/Grag2021_progan/model_epoch_best.pth
#      weights/Grag2021_latent/model_epoch_best.pth
if [ -f "weights/Grag2021_progan/model_epoch_best.pth" ] && [ -f "weights/Grag2021_latent/model_epoch_best.pth" ]; then
    echo "Weights already present, skipping download."
else
    echo "Downloading weights..."
    # gdown 5.0 removed --id; the file id is now a positional argument.
    venv/bin/gdown 1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf -O weights.zip
    mkdir -p weights
    unzip -q weights.zip -d weights_extracted
    rm weights.zip

    # The zip's internal folder layout isn't documented — search for both
    # expected folders wherever they landed and move them into place.
    for name in Grag2021_progan Grag2021_latent; do
        found=$(find weights_extracted -type d -name "$name" | head -n 1)
        if [ -n "$found" ]; then
            mv "$found" "weights/$name"
        else
            echo "WARNING: could not find '$name' inside the downloaded zip."
        fi
    done
    rm -rf weights_extracted

    if [ ! -f "weights/Grag2021_progan/model_epoch_best.pth" ] || [ ! -f "weights/Grag2021_latent/model_epoch_best.pth" ]; then
        echo "WARNING: expected weight files are still missing. Check what's"
        echo "under weights/ and move things into place manually so you have:"
        echo "  weights/Grag2021_progan/model_epoch_best.pth"
        echo "  weights/Grag2021_latent/model_epoch_best.pth"
    fi
fi

echo ""
echo "Setup complete. Run either variant with:"
echo "  models/DMimageDetection/venv/bin/python run.py --model grag2021_progan --dataset <dataset_name>"
echo "  models/DMimageDetection/venv/bin/python run.py --model grag2021_latent  --dataset <dataset_name>"
