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

# Module-level state populated by load(), consumed by predict()
_model = None
_transforms = None


def load():
    """Load RINE weights and prepare the model for inference."""
    global _model, _transforms

    _, _transforms, _ = get_transforms()
    _model = get_our_trained_model(ncls="ldm", device=DEVICE)
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
