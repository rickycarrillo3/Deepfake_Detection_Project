#!/usr/bin/env bash
# Sets up the isolated environment for the gradient-boosting ensemble, per README.md.
# Run from inside gradient_boosting/:
#   cd gradient_boosting && bash setup.sh
#
# Optional: models/aide/ and models/co-spy/ already have scikit-learn + pandas in
# their venvs, so you can skip this and run the trainer under one of those:
#   models/aide/venv/bin/python gradient_boosting/train.py ...
# This venv just avoids depending on which detector happens to be installed.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Interpreter used to build the venv is overridable — on clusters where the
# system python3 is old, load a module and point PYTHON at it, e.g.:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
PYTHON="${PYTHON:-python3}"
echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "Setup complete. From the repo root:"
echo "  gradient_boosting/venv/bin/python gradient_boosting/train.py --list"
echo "  gradient_boosting/venv/bin/python gradient_boosting/train.py \\"
echo "      --models <stem1>,<stem2>,<stem3> --dry-run"
