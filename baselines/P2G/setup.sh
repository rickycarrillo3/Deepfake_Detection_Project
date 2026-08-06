#!/usr/bin/env bash
# Sets up the isolated environment for the P2G baseline, per baselines/README.md.
# Run from inside baselines/P2G/:
#   cd baselines/P2G && bash setup.sh
#
# P2G gets its own venv rather than sharing baselines/venv because it pins
# torch==2.2.2 (and torchvision 0.17.2) — exactly the kind of conflicting pin
# the per-model isolation in models/ exists for.
#
# The checkpoint (best.pt) is NOT downloaded here: it was provided directly by
# the TrueFake authors and is not publicly hosted. The frozen CLIP ViT-B/16
# backbone downloads itself on first run.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# 1. Vendor the upstream repo (src/models/slinet.py, its CLIP fork, and the
#    ImageNet-1k class names used for the zero-shot conditioning pass).
if [ -d "Prompt2Guard" ]; then
    echo "Prompt2Guard/ already exists, skipping clone."
else
    git clone https://github.com/laitifranz/Prompt2Guard.git Prompt2Guard
fi

# 2. Isolated venv. Interpreter is overridable for cluster module systems:
#   module load python/3.12.12 && PYTHON=python3.12 ./setup.sh
#
# Unlike the other five baselines, P2G still pins an older torch, so it cannot
# use just any current interpreter: its pinned torch/torchvision/Pillow publish
# no wheels past cp312, and pip reports that as "no matching distribution found
# for torch==2.2.2" — which reads like a bad pin rather than the interpreter
# being too new. Rather than hardcode a version (see CLAUDE.md), search for a
# usable one when PYTHON is unset, and always let an explicit PYTHON win so a
# cluster's module-loaded interpreter is still what gets used.
#
# Keep this range in sync with requirements.txt: it is the wheel coverage of the
# pins there, not an upstream statement about what P2G supports.
P2G_MAX_MINOR=12  # highest 3.x with torch 2.2.2 / torchvision 0.17.2 / Pillow 10.2.0 wheels
P2G_MIN_MINOR=8

# Prints the minor version of interpreter $1, or nothing if it is missing or
# not Python 3. The `|| true` matters under `set -e`: a nonexistent interpreter
# exits 127, which would otherwise abort the script at the assignment below —
# before the checks that explain what is actually wrong.
python_minor() {
    "$1" -c 'import sys; print(sys.version_info[1] if sys.version_info[0] == 3 else "")' 2>/dev/null || true
}

if [ -n "${PYTHON:-}" ]; then
    minor="$(python_minor "$PYTHON")"
    if [ -z "$minor" ]; then
        echo "ERROR: PYTHON=$PYTHON is not a usable Python 3 interpreter." >&2
        exit 1
    fi
    if [ "$minor" -gt "$P2G_MAX_MINOR" ] || [ "$minor" -lt "$P2G_MIN_MINOR" ]; then
        echo "ERROR: PYTHON=$PYTHON is Python 3.$minor, but P2G needs 3.$P2G_MIN_MINOR-3.$P2G_MAX_MINOR:" >&2
        echo "  its pinned torch/torchvision/Pillow publish no wheels outside that range," >&2
        echo "  so pip would fail with 'no matching distribution found'." >&2
        echo "  Install one and re-run, e.g.: PYTHON=python3.12 bash setup.sh" >&2
        exit 1
    fi
else
    PYTHON=""
    for minor in $(seq "$P2G_MAX_MINOR" -1 "$P2G_MIN_MINOR"); do
        if command -v "python3.$minor" >/dev/null 2>&1; then
            PYTHON="python3.$minor"
            break
        fi
    done
    if [ -z "$PYTHON" ]; then
        default_minor="$(python_minor python3)"
        echo "ERROR: no Python 3.$P2G_MIN_MINOR-3.$P2G_MAX_MINOR interpreter found on PATH." >&2
        echo "  Your default python3 is 3.${default_minor:-?}, and P2G pins torch==2.2.2 /" >&2
        echo "  torchvision==0.17.2 / Pillow==10.2.0, which publish no wheels for it — pip" >&2
        echo "  would fail with 'no matching distribution found for torch==2.2.2'." >&2
        echo "  Install a supported interpreter, e.g.:" >&2
        echo "    brew install python@3.12          # or: conda create -p ./venv python=3.12" >&2
        echo "  then re-run:" >&2
        echo "    PYTHON=python3.12 bash setup.sh" >&2
        exit 1
    fi
    echo "No PYTHON set; auto-selected $PYTHON (P2G needs <= 3.$P2G_MAX_MINOR)."
fi

echo "Building venv with: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

# A venv left behind by an earlier failed run can be on the wrong interpreter,
# which would silently re-fail the same way; check before reusing it below.
if [ -d "venv" ] && [ "$(python_minor venv/bin/python)" != "$(python_minor "$PYTHON")" ]; then
    echo "Existing venv/ is Python 3.$(python_minor venv/bin/python), not 3.$(python_minor "$PYTHON") — removing it."
    rm -rf venv
fi

if [ -d "venv" ]; then
    echo "venv/ already exists, skipping venv creation."
else
    "$PYTHON" -m venv venv
fi

venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo ""
echo "Verifying checkpoint..."
venv/bin/python ../verify_checkpoints.py P2G || {
    echo "Checkpoint verification FAILED — do not trust results from this setup."
    exit 1
}

echo ""
echo "Setup complete. Run with (from repo root):"
echo "  baselines/P2G/venv/bin/python run.py --model tf-p2g --dataset <dataset_name>"
