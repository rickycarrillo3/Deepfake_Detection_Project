import os
import sys

import torch
from PIL import Image
from torchvision import transforms

REPO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreqNet-DeepfakeDetection")
CHECKPOINT_PATH = os.path.join(REPO_DIR, "4-classes-freqnet-v2.pth")


class FreqNetModel:
    """Wrapper for chuangchuangtan/FreqNet-DeepfakeDetection exposing the models/ load()/predict() contract."""

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

    def load(self):
        if REPO_DIR not in sys.path:
            sys.path.insert(0, REPO_DIR)

        if self.device == "cpu":
            # The vendored FreqNet architecture unconditionally calls .cuda() on some
            # parameters in its __init__, assuming a GPU is always present. No-op it so
            # the model can be constructed on CPU-only machines.
            torch.Tensor.cuda = lambda self, *args, **kwargs: self

        from networks.freqnet import freqnet

        self.model = freqnet(num_classes=1)
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        return self

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model(x).sigmoid().flatten().cpu().tolist()[0]
        return float(prob)


_DEFAULT_MODEL = None


def load(**kwargs):
    """Module-level contract: load weights and return a ready FreqNetModel."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = FreqNetModel(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL


def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)
