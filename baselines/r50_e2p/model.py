"""R50-E2P — the TrueFake paper's own detector, as a baselines/ wrapper.

Upstream: https://github.com/MMLab-unitn/TrueFake-IJCNN25 (./detector/).

"ResNet50 as Feature Encoder to Prototypes": a `nodown` ResNet50 (the
Gragnaniello et al. stride-1 first conv) with ImageNet weights frozen, feeding a
trainable 2048->128 fully-connected layer whose output is scored against learned
Gaussian prototypes. Upstream builds this from three flags rather than three
architectures — `--arch nodown --freeze --prototype` — so this wrapper passes the
same flags through a settings shim to upstream's `ImageClassifier`.

The published checkpoint was trained on `gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre`
(StyleGAN2 + StableDiffusion XL fakes vs FFHQ + FORLAB reals, none of it
socially shared), which is why result_stem() reports family `GANDM` rather than
one or the other — see result_stem() below.

Scores: upstream trains with BCEWithLogitsLoss against label Fake=1 and
thresholds the raw output at 0 (`pred > 0`, see upstream train.py
check_accuracy). The output is therefore a logit, not a probability, so predict()
applies a sigmoid — which both puts the score in the [0, 1] this repo's contract
requires and maps upstream's 0 cutoff onto metrics.THRESHOLD (0.5) exactly, so
thresholding here agrees with thresholding upstream.
"""

import os
import sys
import types

import torch
from PIL import Image, ImageFile

# Upstream's dataset.py sets this, and some socially-shared JPEGs in TrueFake are
# in fact truncated. Match it so those images score instead of raising.
ImageFile.LOAD_TRUNCATED_IMAGES = True

HERE = os.path.dirname(os.path.abspath(__file__))
DETECTOR_DIR = os.path.join(HERE, "TrueFake-IJCNN25", "detector")
DEFAULT_CHECKPOINT = os.path.join(HERE, "weights", "best.pt")


class R50E2PModel:
    """Wrapper for MMLab-unitn/TrueFake-IJCNN25's detector exposing the load()/predict() contract."""

    def __init__(self, checkpoint=None, arch="nodown", prototype=True, freeze=True,
                 num_centers=1, device=None):
        self.checkpoint = checkpoint or DEFAULT_CHECKPOINT
        # Strings, because everything arriving via --model-arg is CLI text.
        self.arch = arch
        self.prototype = _as_bool(prototype)
        self.freeze = _as_bool(freeze)
        self.num_centers = int(num_centers)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = None

    def load(self):
        if not os.path.isdir(DETECTOR_DIR):
            raise RuntimeError(
                f"Upstream detector code not found at {DETECTOR_DIR}. Run baselines/r50_e2p/setup.sh first."
            )
        if not os.path.isfile(self.checkpoint):
            raise RuntimeError(
                f"Checkpoint not found at {self.checkpoint}. Run baselines/r50_e2p/setup.sh first."
            )

        if DETECTOR_DIR not in sys.path:
            sys.path.insert(0, DETECTOR_DIR)

        from networks import ImageClassifier
        import torchvision.transforms.v2 as Tv2

        # Upstream's ImageClassifier reads its configuration off the argparse
        # Namespace produced by detector/parser.py. Build the equivalent shim
        # rather than importing that parser, which would try to consume run.py's
        # own argv.
        settings = types.SimpleNamespace(
            arch=self.arch,
            freeze=self.freeze,
            prototype=self.prototype,
            num_centers=self.num_centers,
        )

        self.model = ImageClassifier(settings)
        state_dict = torch.load(self.checkpoint, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        # Upstream's test-split transform (detector/dataset.py transform_start +
        # transform_end), deliberately reproduced in full:
        #   * there is NO resize or crop — the network is fully convolutional and
        #     scores the image at its native resolution. Do not add one here; use
        #     run.py's --resize when a standardized input size is wanted, so the
        #     resize regime is recorded in the CSV rather than baked in silently.
        #   * the FORLAB-specific CenterCrops in upstream's transform_end are
        #     conditioned on `realFORLAB:*` appearing in the data_keys of the run.
        #     They do not apply to this repo's datasets, so they are omitted.
        self.transform = Tv2.Compose(
            [
                Tv2.ToImage(),
                Tv2.ToDtype(torch.float32, scale=True),
                Tv2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        return self

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).squeeze()
            prob = torch.sigmoid(logit).item()
        return float(prob)


def _as_bool(value):
    """--model-arg values arrive as strings; accept the usual spellings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


_DEFAULT_MODEL = None


def load(**kwargs):
    """Module-level contract: load weights and return a ready R50E2PModel."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = R50E2PModel(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL


def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)


def result_stem():
    """Optional runner contract (see run.py): the '<model>_<family>' middle of the
    results filename.

    Family is `GANDM`, not GAN or DM: the one published checkpoint was trained on
    StyleGAN2 *and* StableDiffusion XL fakes together (data_keys
    `gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre`), so neither single-family tag
    would be honest about what it saw.
    """
    return "r50_e2p_GANDM"
