"""P2G adapter — implements the model contract (see baselines/README.md):
    load()               -> load weights, prepare for inference
    predict(image_path)  -> float in [0, 1], predicted fake-probability

Prompt2Guard (Laiti, Liberatori, De Min & Ricci, 2025), "Conditioned
Prompt-Optimization for Continual Deepfake Detection", retrained on TrueFake.

How this one differs from the other five
----------------------------------------
The other baselines are a backbone plus a classifier head. P2G is a
prompt-learning method over frozen CLIP ViT-B/16, so scoring one image takes
*two* CLIP passes:

  1. A zero-shot pass over the ImageNet-1k class names, to find the top-5
     classes the image most resembles. These condition the text prompts:
     the method builds "a {real/fake} photo of a {class}." for each, the
     conditioning that gives the method its name.
  2. The actual detection pass, where the learned prompt pairs are injected
     into both the text and image encoders and the prediction comes from
     cosine similarity between the two sets of embeddings.

Only the learned prompts are trained, which is why best.pt is 28 KB: it holds
`prompt_learner.0.text_prompt` (7x512) and `.img_prompt` (7x768), the frozen
CLIP backbone being downloaded separately.

Configuration is read from the checkpoint, not hardcoded
--------------------------------------------------------
best.pt ships `K`, `topk_classes` and `ensembling_flags` alongside the weights,
so load() reads them from there rather than assuming the upstream defaults.
For the record, the values it carries (K=7, topk_classes=5, ensembling
[False, False, True, False]) match upstream's default configuration —
ensembling applied after the cosine similarity, without confidence-score
weighting.

Single task
-----------
P2G is built for continual learning across a sequence of tasks, and its
`interface()` path uses the checkpoint's `keys` tensors to infer which task an
image belongs to. Here there is exactly one task — the checkpoint has a single
`prompt_learner` entry and `tasks: 0` — so this uses the plain `forward()` path
with `numtask = 1`. The `keys` are consequently unused: with one task the
routing they exist for is degenerate. They are left in the checkpoint rather
than deleted.

Scoring
-------
`forward()` returns logits over `cddb_classnames = {0: "real", 1: "fake"}`, so
the fake-probability is `softmax(logits)[1]` — a genuine 2-class posterior, not
a sigmoid over a single logit like the CNN baselines.

Preprocessing (upstream src/utils/data.py, test_trsf)
------------------------------------------------------
    Resize(256, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)

Note this uses **ImageNet** statistics, not CLIP's, despite the CLIP backbone —
that is what upstream does, and it differs from CLIP-D in this same directory.
"""

import os
import sys

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CLONE_DIR = os.path.join(_MODEL_DIR, "Prompt2Guard")
_SRC_DIR = os.path.join(_CLONE_DIR, "src")
for _path in (_SRC_DIR, os.path.join(_SRC_DIR, "utils")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

WEIGHTS_PATH = os.path.join(_MODEL_DIR, "best.pt")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_model = None
_transform = None
_zeroshot = None  # (text_features, class_names) for the ImageNet-1k pass


def load(**kwargs):
    """Load the CLIP backbone, the learned prompts, and the zero-shot table."""
    global _model, _transform, _zeroshot

    from models.slinet import SliNet
    from models.clip import clip
    from utils.class_names import imagenet1k_classnames

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu")

    args = {
        "dataset": "cddb",
        # One task: the checkpoint holds a single prompt_learner entry.
        "task_name": ["truefake"],
        "K": int(checkpoint["K"]),
        "topk_classes": int(checkpoint["topk_classes"]),
        "ensembling": list(checkpoint["ensembling_flags"]),
        "precision": "fp32",
    }

    model = SliNet(args)
    # SliNet hardcodes self.device = "cuda" in __init__; override so this runs
    # on CPU/MPS too. It is only used to place freshly-built tensors.
    model.device = DEVICE
    model.numtask = 1  # forward() indexes prompt_learner[numtask - 1]

    state_dict = checkpoint["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # Only the prompts were trained; the frozen CLIP backbone is absent from
    # the checkpoint, so `missing` is expected to be all-backbone. What must
    # not happen is a prompt tensor failing to land.
    stray = [k for k in missing if "prompt_learner" in k]
    if stray or unexpected:
        raise RuntimeError(
            f"P2G prompts did not load cleanly: missing={stray}, unexpected={list(unexpected)}"
        )

    _model = model.to(DEVICE).eval()

    # Zero-shot text table, built once: encoding 1000 ImageNet class names per
    # image would otherwise dominate runtime.
    prompts = torch.cat(
        [clip.tokenize(f"a photo of a {name}") for name in imagenet1k_classnames.values()]
    ).to(DEVICE)
    with torch.no_grad():
        text_features = _model.clip_model.encode_text(prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    _zeroshot = (text_features, list(imagenet1k_classnames.values()))

    _transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return _model


def _object_labels(tensor):
    """Top-k ImageNet-1k classes for one image, in the layout SliNet expects.

    Upstream reads these from a precomputed classes.pkl and lets the default
    DataLoader collate reshape them, so `generate_prompts_from_input` receives
    a sequence of topk entries, each a (names-per-batch, scores-per-batch)
    pair. This rebuilds that same layout for a batch of one, computing the
    zero-shot pass live instead of from a pickle — our datasets are not the
    CDDB one that pickle was built for.
    """
    text_features, class_names = _zeroshot

    with torch.no_grad():
        image_features = _model.clip_model.encode_image(tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(_model.topk_classes)

    return [
        [(class_names[idx.item()],), torch.tensor([100.0 * val.item()])]
        for val, idx in zip(values, indices)
    ]


def predict(image_path: str) -> float:
    """Return predicted fake-probability in [0, 1] for a single image."""
    if _model is None or _transform is None:
        raise RuntimeError("model.load() must be called before predict().")

    image = Image.open(image_path).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor, _object_labels(tensor))["logits"]
        # cddb_classnames = {0: "real", 1: "fake"}
        prob_fake = logits.softmax(dim=-1)[0, 1]

    return float(prob_fake)


def result_stem():
    """See run.py. Trained on StyleGAN2 + SDXL fakes, so family GANDM."""
    return "tf-p2g_GANDM"
