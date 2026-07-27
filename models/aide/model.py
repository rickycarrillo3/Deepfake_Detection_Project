import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import transforms

# AIDE's Colab/inference path treats truncated images as loadable where possible.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_DIR = Path(__file__).resolve().parent
REPO_DIR = MODEL_DIR / "AIDE"

# AIDE ships three detector checkpoints, trained on different fake-image
# sources. checkpoint= selects among these by name; checkpoint_path= (or the
# AIDE_WEIGHTS_PATH env var) overrides with an arbitrary path and takes
# priority over checkpoint= if both are given.
CHECKPOINT_FILES = {
    "genimage": "GenImage_train.pth",
    "progan": "progan_train.pth",
    "sd14": "sd14_train.pth",
}
DEFAULT_CHECKPOINT = "genimage"

AIDE_IMAGE_SIZE = 256


class AIDEModel:
    """
    Wrapper around shilinyan99/AIDE exposing the repository's model contract.

    score = predicted fake probability in [0, 1]
          = softmax([real_logit, fake_logit])[1]

    AIDE convention:
      class 0 -> real-like
      class 1 -> fake / AI-generated-like
    """

    def __init__(self, device=None, checkpoint=None, checkpoint_path=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_name = checkpoint or DEFAULT_CHECKPOINT

        if checkpoint_path is not None:
            # Explicit full-path override — wins over checkpoint= entirely.
            self.checkpoint_path = Path(checkpoint_path)
        elif os.environ.get("AIDE_WEIGHTS_PATH"):
            self.checkpoint_path = Path(os.environ["AIDE_WEIGHTS_PATH"])
        else:
            if self.checkpoint_name not in CHECKPOINT_FILES:
                raise ValueError(
                    f"Unknown AIDE checkpoint '{self.checkpoint_name}'. "
                    f"Choose one of: {', '.join(CHECKPOINT_FILES)}"
                )
            self.checkpoint_path = REPO_DIR / "checkpoints" / CHECKPOINT_FILES[self.checkpoint_name]

        self.model = None
        self.dct = None

        self.to_tensor = transforms.ToTensor()
        # Matches the uploaded Colab's AIDE resize/normalize transform.
        self.aide_resize_norm = transforms.Compose(
            [
                transforms.Resize((AIDE_IMAGE_SIZE, AIDE_IMAGE_SIZE)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def load(self):
        if not REPO_DIR.exists():
            raise FileNotFoundError(
                f"Missing vendored AIDE repo at {REPO_DIR}. "
                "Clone it with: git clone https://github.com/shilinyan99/AIDE.git "
                "models/aide/AIDE"
            )

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing AIDE checkpoint '{self.checkpoint_name}' at {self.checkpoint_path}. "
                f"Download the .pth detector checkpoints into models/aide/AIDE/checkpoints/ "
                f"(expects one of: {', '.join(CHECKPOINT_FILES.values())}) "
                "or set AIDE_WEIGHTS_PATH=/path/to/checkpoint.pth"
            )

        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        from models import AIDE as AIDE_module  # type: ignore
        from data.dct import DCT_base_Rec_Module  # type: ignore

        self.dct = DCT_base_Rec_Module(output=AIDE_IMAGE_SIZE)

        # Same construction used in the uploaded Colab notebook.
        self.model = AIDE_module.AIDE(resnet_path=None, convnext_path=None)

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

        # Handle checkpoints saved from DistributedDataParallel.
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if len(missing) > 20:
            print(
                f"Warning: many missing checkpoint keys ({len(missing)}). "
                "This may mean you selected a backbone/pretrain checkpoint instead of an AIDE detector checkpoint."
            )
            print("First missing keys:", missing[:10])
        if len(unexpected) > 20:
            print(f"Warning: many unexpected checkpoint keys ({len(unexpected)}).")
            print("First unexpected keys:", unexpected[:10])

        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            # Safe speedups on Ampere+ GPUs; no-op on older setups if unsupported.
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        return self

    def _preprocess(self, image_path):
        if self.model is None or self.dct is None:
            raise RuntimeError("Call load() before predict().")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_tensor = self.to_tensor(img)  # [3, H, W], values in [0, 1]

        # AIDE's DCT/noise feature extraction. This is CPU-heavy.
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(img_tensor)

        # Same stack as the uploaded Colab:
        # 1. DCT/noise low-rank patch
        # 2. DCT/noise high-rank patch
        # 3. second low-rank patch
        # 4. second high-rank patch
        # 5. resized/normalized RGB image tensor
        x_0 = self.aide_resize_norm(img_tensor)
        x_minmin = self.aide_resize_norm(x_minmin)
        x_maxmax = self.aide_resize_norm(x_maxmax)
        x_minmin1 = self.aide_resize_norm(x_minmin1)
        x_maxmax1 = self.aide_resize_norm(x_maxmax1)

        x = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0)
        return x

    def predict(self, image_path):
        x = self._preprocess(image_path).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(x)
            else:
                logits = self.model(x)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            probs = F.softmax(logits.detach().float().cpu(), dim=1)
            fake_probability = probs[0, 1].item()

        return float(fake_probability)


_DEFAULT_MODEL = None


def load(device=None, checkpoint=None, checkpoint_path=None):
    """Module-level contract: load weights and return a ready AIDEModel.

    checkpoint: one of "genimage" (default), "progan", "sd14" — selects which
    of AIDE's shipped detector checkpoints to use, e.g. via
    --model-arg checkpoint=progan.
    """
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = AIDEModel(device=device, checkpoint=checkpoint, checkpoint_path=checkpoint_path).load()
    return _DEFAULT_MODEL


def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)


def result_stem():
    """Optional runner contract (see run.py): the '<model>_<family>' middle of
    the results filename. AIDE's generation family follows which checkpoint was
    used (--model-arg checkpoint=...): progan is a GAN, sd14 is a diffusion
    model, and genimage (a mixed generator dataset) keeps its own label. The
    differing suffix also keeps the three checkpoints' runs in separate CSVs."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before result_stem().")
    family = {"progan": "GAN", "sd14": "DM"}.get(
        _DEFAULT_MODEL.checkpoint_name, _DEFAULT_MODEL.checkpoint_name
    )
    return f"aide_{family}"