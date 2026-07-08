#!/usr/bin/env bash
# Sets up the isolated environment for models/aeroblade/, per README.md.
# Run from inside models/aeroblade/:
#   cd models/aeroblade && bash setup.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (contains src/aeroblade/, scripts/, setup.py)
if [ -d "aeroblade" ]; then
    echo "aeroblade/ already exists, skipping clone."
else
    git clone https://github.com/jonasricker/aeroblade.git aeroblade
fi

# 2. Isolated venv, Python 3.10 (upstream is tested against this version)
if [ -d "aeroblade_env" ]; then
    echo "aeroblade_env/ already exists, skipping venv creation."
else
    python3.10 -m venv aeroblade_env
fi

aeroblade_env_pip="aeroblade_env/bin/pip"
"$aeroblade_env_pip" install --upgrade pip
"$aeroblade_env_pip" install -r aeroblade/requirements.txt
"$aeroblade_env_pip" install -e ./aeroblade
"$aeroblade_env_pip" install -r requirements.txt

echo ""
echo "Setup complete. First run will download AE weights from HuggingFace"
echo "(CompVis/stable-diffusion-v1-1, stabilityai/stable-diffusion-2-base,"
echo "kandinsky-community/kandinsky-2-1) — make sure you're logged in via"
echo "'aeroblade_env/bin/huggingface-cli login' if any require accepting a license."
echo ""
echo "Run with:"
echo "  PYTHONPATH=<repo_root> aeroblade_env/bin/python run.py <dataset_name>"
