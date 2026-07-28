"""NPR adapter — implements the model contract (see baselines/README.md):
    load()               -> load weights, prepare for inference
    predict(image_path)  -> float in [0, 1], predicted fake-probability

Neighboring Pixel Relationships (Tan et al., CVPR 2024, "Rethinking the
Up-Sampling Operations in CNN-Based Generative Network for Generalizable
Deepfake Detection"), retrained on TrueFake by the paper's authors.

The architecture is imported from the vendored upstream clone rather than
retranscribed: `networks.resnet.resnet50(num_classes=1)`, which despite the
name is a *truncated* ResNet-50 — a 3x3 stride-2 stem, only layer1 and layer2,
and `fc1: 512 -> 1`. Its forward computes the NPR residual internally
(`x - interpolate(x, 0.5)`, i.e. the image minus its nearest-neighbour
down/up-sampled reconstruction), so predict() must hand it a plain normalized
image and must *not* pre-compute the residual.

Preprocessing
-------------
ToTensor + ImageNet normalization at native resolution. Upstream's test
options set no_resize/no_crop, and resizing would in any case corrupt the
up-sampling artifacts NPR exists to detect.

Checkpoint format
-----------------
Upstream's validate.py reads `state_dict['model']`, but this checkpoint was
saved bare (its top-level keys are the tensors themselves), so both layouts
are accepted.
"""

import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CLONE_DIR = os.path.join(_MODEL_DIR, "NPR-DeepfakeDetection")
if _CLONE_DIR not in sys.path:
    sys.path.insert(0, _CLONE_DIR)

WEIGHTS_PATH = os.path.join(_MODEL_DIR, "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def load(**kwargs):
    """Load the NPR checkpoint and prepare it for inference."""
    global _model, _transform

    from networks.resnet import resnet50

    model = resnet50(num_classes=1)
    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)
    _model = model.to(DEVICE).eval()

    _transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return _model


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transform is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = _model(tensor).flatten()[0]

    return float(torch.sigmoid(logit))


def result_stem():
    """See run.py. Trained on StyleGAN2 + SDXL fakes, so family GANDM."""
    return "tf-npr_GANDM"
