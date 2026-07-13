"""
CNNDetection model adapter — implements the models/<model_name>/ contract:
    load()                    -> load weights, prepare model for inference
    predict(image_path)       -> float in [0, 1], predicted fake-probability

Wraps PeterWang512/CNNDetection (vendored in ./CNNDetection/), the pretrained
ResNet-50 classifier from "CNN-generated images are surprisingly easy to
spot...for now" (CVPR 2020), using its networks/resnet.py::resnet50 and the
blur+JPEG-augmented weights (weights/blur_jpg_prob0.5.pth).

Preprocessing / architecture notes (from the repo's demo.py / demo_dir.py)
--------------------------------------------------------------------------
- resnet50(num_classes=1): a standard ResNet-50 whose final fc outputs a
  single logit. The checkpoint is a dict; the weights live under its
  'model' key (state_dict['model']).
- No resize/crop by default. demo.py exposes an optional CenterCrop
  (CROP_SIZE, e.g. 224) but the paper/demo default is no crop, so images
  are scored at native resolution. Set CROP_SIZE below to change this.
- Normalize with ImageNet mean/std after ToTensor(), matching the repo.
- The logit is a genuine binary-classifier logit (positive = fake), so
  sigmoid() gives a real per-image probability that doesn't depend on
  anything else in the batch/dataset.

Weight variant
--------------
weights/download_weights.sh fetches two checkpoints, trained with
different blur+JPEG augmentation probabilities:
    blur_jpg_prob0.5.pth  (heavier aug — the demo default; used here)
    blur_jpg_prob0.1.pth
Point WEIGHTS_PATH at the other file to switch.
"""

import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.join(_MODEL_DIR, "CNNDetection")
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

WEIGHTS_PATH = os.path.join(_REPO_DIR, "weights", "blur_jpg_prob0.5.pth")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# None -> no crop (paper/demo default). Set to e.g. 224 to center-crop.
CROP_SIZE = None

_model = None
_transform = None


def load():
    """Load CNNDetection weights and prepare the model for inference."""
    global _model, _transform

    from networks.resnet import resnet50  # from the vendored repo

    model = resnet50(num_classes=1)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    _model = model.to(DEVICE).eval()

    crop = [transforms.CenterCrop(CROP_SIZE)] if CROP_SIZE is not None else []
    _transform = transforms.Compose(
        crop
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transform is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = _model(tensor).sigmoid().flatten().cpu().tolist()[0]

    return float(prob)
