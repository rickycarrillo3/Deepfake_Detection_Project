"""
Grag2021_progan model adapter — implements the models/<model_name>/ contract:
    load()                    -> load weights, prepare model for inference
    predict(image_path)       -> float in [0, 1], predicted fake-probability

Wraps grip-unina/DMimageDetection (vendored in ../dmimagedetection/,
shared with the sibling grag2021_latent/ model — both variants use the
same clone and the same downloaded weights zip), specifically
test_code/get_method_here.py's get_method_here()/def_model() and
test_code/normalization.py's get_list_norm(), which is what
test_code/main.py itself calls. (The repo's inference code lives under
test_code/, not the repo root — training_code/ and docs/ are siblings.)

This is one of two variants of the same architecture trained on
different data — see ../grag2021_latent/ for the sibling ('Grag2021_latent',
trained on latent-diffusion outputs vs. this one's ProGAN training).
Kept as separate model directories since they're independent detectors
worth correlating separately, not one model with two settings.

Preprocessing / architecture notes (from get_method_here.py + main.py)
------------------------------------------------------------------------
- arch='res50stride1': a fully-convolutional ResNet-50 (stride0=1,
  gap_size=1 -- see networks/networks/resnet_mod.py) with NO adaptive
  pooling to a fixed size. Output is a per-pixel logit map, not a single
  scalar.
- norm_type='resnet', patch_size=None: no resize/crop at all — just
  ToTensor() + ImageNet mean/std normalization on the image at its
  native resolution. (Unlike RINE, there's no fixed input size here.)
- main.py's reduction: out.shape[1]==1 -> drop that channel -> average
  the remaining (H, W) spatial map into a single logit per image. We
  replicate that exactly in predict() below.
- The logit is a genuine binary-classifier logit (positive = fake per
  the paper's README), so sigmoid() here gives a real per-image
  probability — unlike AEROBLADE's raw distances, this doesn't depend
  on anything else in the batch/dataset.
"""

import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # models/DMimageDetection/
_REPO_DIR = os.path.join(_PARENT_DIR, "dmimagedetection", "test_code")
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from get_method_here import get_method_here, def_model  # noqa: E402
from normalization import get_list_norm  # noqa: E402

MODEL_NAME = "Grag2021_progan"
WEIGHTS_DIR = os.path.join(_PARENT_DIR, "weights")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def _patch_networks_double_import():
    """
    Work around a mismatch between get_method_here.py's def_model(), which
    does `import networks.networks.resnet_mod` (double-nested), and the
    actual cloned repo, which only has a single-level `networks` package
    (test_code/networks/resnet_mod.py directly, no nested networks/networks/
    folder). Rather than patch the vendored file, alias the module in
    sys.modules so the import resolves as written. Idempotent — safe to
    call from both load() functions if this repo is ever shared further.
    """
    if "networks.networks" in sys.modules:
        return
    import networks  # test_code/networks/__init__.py
    import networks.resnet_mod as _resnet_mod

    sys.modules["networks.networks"] = networks
    sys.modules["networks.networks.resnet_mod"] = _resnet_mod


def load():
    """Load Grag2021_progan weights and prepare the model for inference."""
    global _model, _transform

    _patch_networks_double_import()

    _, model_path, arch, norm_type, patch_size = get_method_here(
        MODEL_NAME, weights_path=WEIGHTS_DIR
    )
    # Both Grag2021_progan/latent use patch_size=None (no resize/crop). If a
    # future weights update changes this, the transform below needs updating
    # to match get_method_here.py's crop/resize branches.
    assert patch_size is None, (
        f"unexpected patch_size {patch_size!r} for {MODEL_NAME}; "
        "update model.py's preprocessing to match get_method_here.py"
    )

    _model = def_model(arch, model_path, localize=False)
    _model = _model.to(DEVICE).eval()

    _transform = transforms.Compose(get_list_norm(norm_type))


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transform is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = _model(tensor).cpu().numpy()

    # Fully-convolutional output: (1, 1, H, W) -> drop channel -> spatial mean.
    # Mirrors main.py's out_tens[:, 0] then np.mean(out_tens, (1, 2)) exactly.
    if out.shape[1] == 1:
        out = out[:, 0]
    logit = float(np.mean(out))

    return float(torch.sigmoid(torch.tensor(logit)))


def result_stem():
    """Optional runner contract (see run.py): the '<model>_<family>' middle of
    the results filename. This DMImageDetection checkpoint was trained on
    ProGAN (GAN) images."""
    return "grag2021_progan_GAN"