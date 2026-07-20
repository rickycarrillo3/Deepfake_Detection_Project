"""
RINE model adapter — implements the models/<model_name>/ contract:
    load()                    -> load weights, prepare model for inference
    predict(image_path)       -> float in [0, 1], predicted fake-probability

Wraps mever-team/rine (vendored in ./rine/), specifically the
`get_transforms` / `get_our_trained_model` helpers used in the original
demo.py / test.py scripts.

Assumes this file's own venv (../rine_env or similar, created per the
model's requirements.txt) is the interpreter running it — torch==2.1.1 /
torchvision==0.16.1 / numpy==1.26.4, matching the pins in the source notebook.
"""

import os
import sys

import torch
from PIL import Image

# The vendored rine repo (git clone https://github.com/mever-team/rine)
# is expected to live alongside this file, at models/rine/rine/
_RINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rine")
if _RINE_DIR not in sys.path:
    sys.path.insert(0, _RINE_DIR)

from src.utils import get_transforms, get_our_trained_model  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# RINE ships checkpoints for multiple training configurations in ckpt/ —
# model_1class_trainable.pth, model_2class_trainable.pth,
# model_4class_trainable.pth, and model_ldm_trainable.pth (the original
# default here). get_our_trained_model()'s ncls argument selects among them,
# and — confirmed from src/utils.py — checks it as the INTEGER 1, 2, or 4, or
# the literal string "ldm"; there is no other string form. Since --model-arg
# always passes values through as strings, load() below coerces back to int
# for anything other than "ldm".
DEFAULT_NCLS = "ldm"

# Module-level state populated by load(), consumed by predict()
_model = None
_transforms = None
_ncls = None


def load(ncls=None):
    """Load RINE weights and prepare the model for inference.

    ncls: which of RINE's shipped checkpoints to use — "ldm" (default,
    matching the prior hardcoded behavior) or 1, 2, 4. Passed straight
    through to get_our_trained_model, coerced to int unless it's "ldm"
    (get_our_trained_model checks ncls==1/2/4 as integers, and --model-arg
    always hands this function a string). Set via --model-arg ncls=<value>.
    """
    global _model, _transforms, _ncls

    raw = ncls if ncls is not None else DEFAULT_NCLS
    _ncls = raw if isinstance(raw, str) and raw.lower() == "ldm" else int(raw)

    _, _transforms, _ = get_transforms()
    _model = get_our_trained_model(ncls=_ncls, device=DEVICE)
    _model.to(DEVICE)
    _model.eval()


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transforms is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    tensor = _transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = _model(tensor)[0]
        probability = torch.sigmoid(logit)

    return float(probability.detach().cpu().numpy()[0][0])


def result_stem():
    """Optional runner contract (see run.py): the '<model>_<family>' middle of
    the results filename. RINE was trained on latent-diffusion data, so its
    generation family is always DM; the ncls checkpoint (--model-arg ncls=...)
    goes in the model slot (rineldm, rine1, rine2, rine4) so the four
    checkpoints' runs don't overwrite each other's CSVs."""
    if _ncls is None:
        raise RuntimeError("Call load() before result_stem().")
    return f"rine{_ncls}_DM"