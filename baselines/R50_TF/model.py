"""R50-E2P adapter — implements the model contract (see baselines/README.md):
    load()               -> load weights, prepare for inference
    predict(image_path)  -> float in [0, 1], predicted fake-probability

"ResNet50 as Feature Encoder to Prototypes", the detector proposed by the
TrueFake paper itself (§IV.6): a no-downsample ResNet-50 producing a 128-d
embedding, classified by a learned-prototype head instead of a linear layer.
The checkpoint directory is named R50_TF as uploaded; R50-E2P is the paper's
name for it.

Provenance
----------
Architecture and preprocessing are transcribed from the authors' own training
code, MMLab-unitn/TrueFake-IJCNN25 (detector/networks.py, detector/dataset.py),
linked from footnote 4 of the paper. Not inferred.

Preprocessing (detector/dataset.py, split == 'test')
----------------------------------------------------
    ToImage() -> ToDtype(float32, scale=True) -> Normalize(ImageNet mean/std)

There is deliberately **no resize and no crop**: the test transform is only
those three steps. (The CenterCrop entries in the authors' test pipeline are
guarded by `'realFORLAB:...' in data_keys` and so apply only to their FORLAB
real-image subset, not to FFHQ or any generated set.) Images are therefore
scored at native resolution, and because arch='nodown' removes the stride from
the stem, compute scales with the full image area.

Scoring
-------
train.py optimizes the head with BCEWithLogitsLoss, so the prototype score is
a logit, not a probability — the authors' own test.py writes it out raw and
unbounded. sigmoid() maps it to the [0, 1] this project's contract requires;
being monotonic, thresholding at 0.5 here is equivalent to thresholding their
raw score at 0.
"""

import os
import sys

import torch
import torchvision.transforms.v2 as Tv2
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # matches the authors' dataset.py

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_BASELINES_DIR = os.path.dirname(_MODEL_DIR)
if _BASELINES_DIR not in sys.path:
    sys.path.insert(0, _BASELINES_DIR)

WEIGHTS_PATH = os.path.join(_MODEL_DIR, "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None


def load(**kwargs):
    """Load the R50-E2P checkpoint and prepare it for inference."""
    global _model, _transform

    from tf_networks import ImageClassifier

    model = ImageClassifier(arch="nodown", prototype=True, num_centers=1)
    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    # strict: a silently-partial load here would still produce plausible
    # scores, so let an architecture mismatch fail loudly instead.
    model.load_state_dict(state_dict, strict=True)
    _model = model.to(DEVICE).eval()

    _transform = Tv2.Compose(
        [
            Tv2.ToImage(),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    """See run.py. Trained on StyleGAN2 *and* SDXL fakes (paper §V), so the
    generation family is both — recorded as GANDM rather than GAN or DM. The
    tf- prefix marks these as the TrueFake authors' retrained baselines, to
    keep them distinct from the original-author checkpoints under models/."""
    return "tf-r50e2p_GANDM"
