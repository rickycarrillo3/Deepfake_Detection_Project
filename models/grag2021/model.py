import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GANimageDetection")
WEIGHTS_PATH = os.path.join(REPO_DIR, "weights", "gandetection_resnet50nodown_stylegan2.pth")

EVAL_IMAGE_SIZE = 256


class Grag2021Model:
    """Wrapper for grip-unina/GANimageDetection exposing the models/ load()/predict() contract."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (EVAL_IMAGE_SIZE, EVAL_IMAGE_SIZE),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load(self):
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)
        from resnet50nodown import resnet50nodown

        self.model = resnet50nodown(self.device, WEIGHTS_PATH)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        return self

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            x = x.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            if self.device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logit = self.model(x).flatten()[0]
            else:
                logit = self.model(x).flatten()[0]

        logit = float(logit.detach().float().cpu())
        # Model convention: negative logit means fake-like. Convert to fake-probability score.
        return float(1 / (1 + np.exp(logit)))


_DEFAULT_MODEL = None


def load(**kwargs):
    """Module-level contract: load weights and return a ready Grag2021Model."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = Grag2021Model(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL


def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)
