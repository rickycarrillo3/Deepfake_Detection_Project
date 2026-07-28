"""MISLNet adapter — implements the model contract (see baselines/README.md):
    load()               -> load weights, prepare for inference
    predict(image_path)  -> float in [0, 1], predicted fake-probability

MISLNet (Bayar & Stamm's constrained-convolution CNN, as applied to synthetic
media by Vahdati et al., CVPRW 2024) retrained on TrueFake by the paper's
authors.

Provenance / how much of this is inferred
-----------------------------------------
This is the least directly-sourced of the six baselines, and worth reading
before trusting its numbers. The Stamm-lab code is not public, and the TrueFake
authors' released repo (MMLab-unitn/TrueFake-IJCNN25) contains only their two
ResNet-50 variants — not MISLNet. So the architecture in baselines/tf_networks.py
is reconstructed from the checkpoint's own tensor shapes.

What is *verified* rather than assumed:
  - every layer shape is forced by the checkpoint, and the whole module loads
    with strict=True (see verify_checkpoints.py);
  - `flatten_index_permutation` is read from the checkpoint (it is [0,1,2,3],
    i.e. plain NCHW) rather than guessed;
  - the constrained conv is renormalized in forward, because the checkpoint's
    stored kernels are demonstrably unconstrained — see ConstrainedConv2d.

What remains genuinely inferred:
  - the 96x96 input size, derived from fc_blocks.0 being (200, 4608) =
    128 x 6 x 6 under SAME padding with three stride-2 pools. It is the only
    clean solution and it agrees with the paper, which says MISLNet was trained
    with R50-E2P's augmentation — a pipeline ending in RandomCrop(96).
  - tanh activations and pooling placement, which follow the standard MISLNet
    topology.

Patch aggregation
-----------------
96x96 is far smaller than the images under test, so a single patch would throw
away almost all of the image and make the score depend on where the crop
landed. Instead the image is tiled into a grid of non-overlapping 96x96 patches
and the per-patch fake-probabilities are averaged — the usual way this family
of patch-based forensic CNNs is applied at full resolution. Set AGGREGATE to
"center" to score a single center crop instead.

Scoring
-------
The head is 2-class (real, fake), not a single logit, so this takes
softmax(...)[1] rather than a sigmoid.
"""

import os
import sys

import torch
import torchvision.transforms.v2 as Tv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINES_DIR = os.path.dirname(_MODEL_DIR)
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

WEIGHTS_PATH = os.path.join(_MODEL_DIR, "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PATCH_SIZE = 96
AGGREGATE = "grid"  # "grid" (mean over tiled patches) or "center"
MAX_PATCHES = 256  # cap so one huge image can't dominate runtime
BATCH_SIZE = 64

_model = None
_transform = None


def load(**kwargs):
    """Load the MISLNet checkpoint and prepare it for inference."""
    global _model, _transform

    from tf_networks import MISLNet

    model = MISLNet(num_classes=2)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    _model = model.to(DEVICE).eval()

    # Same normalization as the R50 baselines: MISLNet was retrained by the
    # TrueFake authors inside their pipeline, whose transform_end is ImageNet
    # ToDtype+Normalize (detector/dataset.py).
    _transform = Tv2.Compose(
        [
            Tv2.ToImage(),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return _model


def _patches(tensor):
    """Tile a [3, H, W] tensor into non-overlapping PATCH_SIZE patches.

    Images smaller than one patch are reflect-padded up to it, so a small
    input still yields exactly one valid patch rather than being skipped.
    """
    _, height, width = tensor.shape

    if height < PATCH_SIZE or width < PATCH_SIZE:
        pad_h = max(0, PATCH_SIZE - height)
        pad_w = max(0, PATCH_SIZE - width)
        tensor = torch.nn.functional.pad(
            tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
        ).squeeze(0)
        _, height, width = tensor.shape

    if AGGREGATE == "center":
        top = (height - PATCH_SIZE) // 2
        left = (width - PATCH_SIZE) // 2
        return tensor[:, top : top + PATCH_SIZE, left : left + PATCH_SIZE].unsqueeze(0)

    patches = (
        tensor.unfold(1, PATCH_SIZE, PATCH_SIZE)
        .unfold(2, PATCH_SIZE, PATCH_SIZE)
        .reshape(3, -1, PATCH_SIZE, PATCH_SIZE)
        .permute(1, 0, 2, 3)
    )
    return patches[:MAX_PATCHES]


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transform is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    patches = _patches(_transform(image))

    probs = []
    with torch.no_grad():
        for start in range(0, len(patches), BATCH_SIZE):
            batch = patches[start : start + BATCH_SIZE].to(DEVICE)
            probs.append(_model(batch).softmax(dim=1)[:, 1].cpu())

    return float(torch.cat(probs).mean())


def result_stem():
    """See run.py. Trained on StyleGAN2 + SDXL fakes, so family GANDM."""
    return "tf-mislnet_GANDM"
