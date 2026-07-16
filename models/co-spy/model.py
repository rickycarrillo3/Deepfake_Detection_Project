import importlib
import importlib.util
import os
import sys
import types
import torch
from PIL import Image
COSPY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CO-SPY")
class CoSpyModel:
    """Wrapper for Megum1/CO-SPY exposing the models/ load()/predict() contract."""
    def __init__(self, train_dataset="sd-v1_4", device=None):
        self.train_dataset = train_dataset
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = None
    def load(self):
        sys.path = [p for p in sys.path if p != COSPY_DIR]
        sys.path.insert(0, COSPY_DIR)
        importlib.invalidate_caches()
        if "evaluate" in sys.modules:
            del sys.modules["evaluate"]
        spec = importlib.util.spec_from_file_location(
            "evaluate",
            os.path.join(COSPY_DIR, "evaluate.py"),
            submodule_search_locations=[COSPY_DIR],
        )
        cospy_eval_mod = importlib.util.module_from_spec(spec)
        sys.modules["evaluate"] = cospy_eval_mod
        spec.loader.exec_module(cospy_eval_mod)
        # With pretrain=True, CO-SPY's Detector ignores its ckpt argument and resolves weights
        # from the *relative* path "pretrained/<train_dataset>/<branch>_weights.pth" — fine in
        # the notebook this was ported from, where the CWD was CO-SPY/, but run.py imports us
        # from the repo root, where that path does not exist. Score the constructor from inside
        # COSPY_DIR so the relative lookup lands on models/co-spy/CO-SPY/pretrained/.
        # (pretrain=False is not an escape: it expects <branch>/best_model.pth, which is not
        # what the pretrained checkpoints are named.)
        prev_cwd = os.getcwd()
        os.chdir(COSPY_DIR)
        try:
            self.detector = cospy_eval_mod.Detector(
                device=self.device,
                train_dataset=self.train_dataset,
                pretrain=True,
            )
        finally:
            os.chdir(prev_cwd)
        self._patch_sem_forward()
        return self
    def _patch_sem_forward(self):
        # Only the progan checkpoint needs patching. Its semantic branch is a HuggingFace
        # transformers CLIPModel, whose forward the installed transformers version returns as a
        # model-output object rather than a raw tensor, and whose multimodal signature needs a
        # dummy input_ids. The sd-v1_4 branch is a different model family — open_clip SigLIP,
        # whose upstream forward already calls self.clip.encode_image(x) and returns a plain
        # tensor. Patching it to call self.clip(x) instead returns open_clip's
        # (image_features, text_features, logit_scale) tuple, which then hits self.fc as a tuple
        # ("linear(): argument 'input' must be Tensor, not tuple"), so leave it alone.
        if self.train_dataset != "progan":
            return
        model = self.detector.model
        if not hasattr(model, "sem"):
            return
        def patched_forward(self_, x, return_feat=False):
            dummy_input_ids = torch.zeros((x.shape[0], 2), dtype=torch.long, device=x.device)
            feat = self_.clip(pixel_values=x, input_ids=dummy_input_ids)
            if not isinstance(feat, torch.Tensor):
                feat = getattr(feat, "image_embeds", getattr(feat, "pooler_output", feat))
            out = self_.fc(feat)
            return (feat, out) if return_feat else out
        model.sem.forward = types.MethodType(patched_forward, model.sem)
    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.detector.model.test_transform(image).unsqueeze(0).to(self.device)
        return float(self.detector.model.predict(image)[0])
_DEFAULT_MODEL = None
def load(**kwargs):
    """Module-level contract: load weights and return a ready CoSpyModel."""
    global _DEFAULT_MODEL
    _DEFAULT_MODEL = CoSpyModel(**kwargs)
    _DEFAULT_MODEL.load()
    return _DEFAULT_MODEL
def predict(image_path):
    """Module-level contract: predict fake-probability for one image path."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before predict().")
    return _DEFAULT_MODEL.predict(image_path)
def result_subdir():
    """Optional runner contract (see run.py): split results by which pretrained
    checkpoint was used, since --model-arg train_dataset=... changes the weights
    entirely and progan/sd-v1_4 runs shouldn't overwrite each other's CSVs."""
    if _DEFAULT_MODEL is None:
        raise RuntimeError("Call load() before result_subdir().")
    return f"co-spy-{_DEFAULT_MODEL.train_dataset}"