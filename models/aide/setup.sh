#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d "AIDE" ]; then
  git clone https://github.com/shilinyan99/AIDE.git AIDE
fi

# ---------------------------------------------------------------------------
# Apply our patches to the vendored clone. The clone is git-ignored, so these
# cannot be tracked as source and must be reapplied to whatever is on disk.
# Idempotent — see patch_vendored.py for what each patch does and why. Run it
# directly against a clone that predates a new patch:
#   python3 models/aide/patch_vendored.py
# ---------------------------------------------------------------------------
python3 patch_vendored.py

# Interpreter used to build the venv. Override on clusters where the system
# python3 is old, e.g. after `module load python/3.12.12`:
#   PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

"$PYTHON" -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt
pip install torch torchvision

# ---------------------------------------------------------------------------
# Download pretrained checkpoints from the AIDE authors' Google Drive folder.
# The folder contains checkpoints for several training sets (GenImage, etc).
# We pull the whole folder with gdown (already installed above via
# requirements.txt) and drop it under AIDE/checkpoints/.
#
# Skip this step with:
#   SKIP_CHECKPOINT_DOWNLOAD=1 ./setup.sh
# ---------------------------------------------------------------------------
CHECKPOINT_DIR="AIDE/checkpoints"
CHECKPOINT_FOLDER_URL="https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4"

if [ "${SKIP_CHECKPOINT_DOWNLOAD:-0}" = "1" ]; then
  echo
  echo "SKIP_CHECKPOINT_DOWNLOAD=1 set — skipping checkpoint download."
elif [ -d "$CHECKPOINT_DIR" ] && [ -n "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
  echo
  echo "$CHECKPOINT_DIR already has files in it — skipping checkpoint download."
  echo "(delete it, or set SKIP_CHECKPOINT_DOWNLOAD=0 and remove existing files, to re-fetch)"
else
  echo
  echo "Downloading AIDE pretrained checkpoints from Google Drive..."
  mkdir -p "$CHECKPOINT_DIR"
  # No --quiet: it suppresses gdown's per-file progress bars, leaving the (slow) folder
  # download looking hung with no indication of progress.
  if gdown --folder "$CHECKPOINT_FOLDER_URL" -O "$CHECKPOINT_DIR"; then
    echo "Checkpoint download complete. Contents of $CHECKPOINT_DIR:"
    ls -la "$CHECKPOINT_DIR"
  else
    echo "WARNING: gdown failed to download the checkpoint folder (Google Drive"
    echo "sometimes rate-limits anonymous downloads of shared folders). You can"
    echo "retry manually with:"
    echo "  source models/aide/venv/bin/activate"
    echo "  gdown --folder '$CHECKPOINT_FOLDER_URL' -O $CHECKPOINT_DIR"
    echo "or open the link in a browser and place the .pth file(s) at:"
    echo "  models/aide/$CHECKPOINT_DIR/"
  fi
fi

echo
echo "Test from the repo root with:"
echo "  source models/aide/venv/bin/activate"
echo "  models/aide/venv/bin/python run.py --model aide --dataset flux_faces --force-label 1 --limit 10"
echo
echo "For the full fake-only run:"
echo "  models/aide/venv/bin/python run.py --model aide --dataset flux_faces --force-label 1"
