"""CLIP-D adapter — implements the model contract (see baselines/README.md):
    load()               -> load weights, prepare for inference
    predict(image_path)  -> float in [0, 1], predicted fake-probability

The CLIP-based detector of Cozzolino et al., CVPRW 2024 ("Raising the Bar of
AI-generated Image Detection with CLIP"): a frozen CLIP ViT used as a semantic
feature extractor with a single trainable linear layer on top. Retrained on
TrueFake by the paper's authors, so only that linear layer ships in best.pt
(`fc.weight` is 1x1024, `fc.bias` a scalar — the whole checkpoint is 15 KB).

Which backbone? — determined, not guessed
------------------------------------------
The checkpoint alone cannot say, so this follows the upstream repo
(grip-unina/ClipBased-SyntheticImageDetection). Both of its shipped CLIP
detectors — weights/clipdet_latent10k/config.yaml and clipdet_latent10k_plus —
specify:

    arch: opencliplinearnext_clipL14commonpool
    norm_type: clip
    patch_size: Clip224

which resolves to `OpenClipLinear(pretrain='clipL14commonpool',
next_to_last=True)`. The `next_to_last` part is what makes the numbers line up:
it sets `backbone.visual.proj = None` and takes the ViT-L-14 features
*before* the output projection, giving **1024** dims rather than the projected
768. That is exactly the checkpoint's fc input width, so the 1024 confirms the
next-to-last ViT-L-14 configuration.

Note this means the more familiar 768-d Ojha/UniversalFakeDetect setup would
be the wrong reading of this checkpoint.

`pretrain` is exposed as a --model-arg for the record, since every ViT-L-14
variant has the same 1024 width and so the checkpoint cannot distinguish
them; `clipL14commonpool` is the upstream default and what both configs use.

    --model-arg pretrain=clipL14openai

Preprocessing (upstream main.py, patch_size == 'Clip224')
---------------------------------------------------------
    Resize(224, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize(CLIP stats)

CLIP's mean/std, **not** ImageNet's — the other five baselines here use
ImageNet stats, so this one deliberately differs. Features are L2-normalized
by `encode_image(..., normalize=True)` before the linear layer.

Unlike the CNN baselines this one does resize to 224, so --resize interacts
with it only as a prior downscale.
"""

import os
import sys

import torch
from PIL import Image, ImageFile
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize
from torchvision.transforms import Resize, ToTensor

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CLONE_DIR = os.path.join(_MODEL_DIR, "ClipBased-SyntheticImageDetection")
if _CLONE_DIR not in sys.path:
    sys.path.insert(0, _CLONE_DIR)

WEIGHTS_PATH = os.path.join(_MODEL_DIR, "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def load(pretrain="clipL14commonpool", **kwargs):
    """Load the CLIP backbone plus the trained linear head."""
    global _model, _transform

    from networks.openclipnet import OpenClipLinear

    model = OpenClipLinear(num_classes=1, pretrain=pretrain, normalize=True, next_to_last=True)

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    # Only the linear head was trained; the frozen backbone is not in the
    # checkpoint (and OpenClipLinear deliberately hides it from state_dict by
    # holding it in a list). strict=True therefore applies to the head alone,
    # which is what we want to verify.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [k for k in unexpected]
    missing = [k for k in missing if k.startswith("fc.")]
    if missing or unexpected:
        raise RuntimeError(
            f"CLIP-D head did not load cleanly: missing={missing}, unexpected={unexpected}"
        )

    _model = model.to(DEVICE).eval()

    _transform = Compose(
        [
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
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
    return "tf-clipd_GANDM"
