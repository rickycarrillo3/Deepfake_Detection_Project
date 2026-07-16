# models/co-spy/

Wrapper around [Megum1/CO-SPY](https://github.com/Megum1/CO-SPY), a semantic+artifact+fusion
deepfake detector. Ported from the `Co-Spy_Setup.ipynb` Colab exploration.

## Setup

```
./models/co-spy/setup.sh
```

This clones `CO-SPY`, creates `models/co-spy/venv/`, and installs `requirements.txt`. It does
**not** install `torch`/`torchvision` (install those separately, matched to your local CUDA
version) or download pretrained checkpoints.

Download pretrained checkpoints with CO-SPY's own script. It `wget`s and unzips into the *current*
directory, so `cd` into `pretrained/` first — running it as `sh pretrained/download.sh` from the
CO-SPY root unpacks the weights one level too high, where the loader will not find them:

```
cd models/co-spy/CO-SPY/pretrained && sh download.sh
```

That should leave each `<train_dataset>` name with:

```
models/co-spy/CO-SPY/pretrained/<train_dataset>/semantic_weights.pth
models/co-spy/CO-SPY/pretrained/<train_dataset>/artifact_weights.pth
models/co-spy/CO-SPY/pretrained/<train_dataset>/fusion_weights.pth
```

`<train_dataset>` selects which fusion checkpoint set to load (e.g. `progan`, `sd-v1_4`) — it is
not the same as the `<dataset_name>` being evaluated from `data/`.

## Known quirks (carried over from the notebook)

- The two checkpoints use different CLIP families in their semantic branch. `progan` uses a
  HuggingFace `transformers` `CLIPModel`; the version we install returns a model-output object from
  its forward instead of a raw tensor, and its multimodal signature needs a dummy `input_ids`
  alongside `pixel_values`. `model.py` monkey-patches `detector.model.sem.forward` at load time to
  paper over both — but **only for `progan`** (see `_patch_sem_forward`). `sd-v1_4` uses `open_clip`
  SigLIP, whose upstream forward (`self.clip.encode_image(x)`) already returns a tensor; patching it
  to call `self.clip(x)` instead yields open_clip's `(image, text, logit_scale)` tuple, which then
  fails in `fc` with `linear(): argument 'input' must be Tensor, not tuple`, so it is left as-is.
- With `pretrain=True`, CO-SPY's `Detector` ignores its `ckpt` argument and loads weights from the
  *relative* path `pretrained/<train_dataset>/<branch>_weights.pth`. That assumed the notebook's
  CWD (`CO-SPY/`), but `run.py` imports `model.py` from the repo root, so the lookup missed and
  loading failed. `load()` therefore constructs the `Detector` from inside `COSPY_DIR` and restores
  the CWD afterwards. `pretrain=False` is not an alternative — it expects `<branch>/best_model.pth`,
  which is not how the pretrained checkpoints are named.
- Do not pin `numpy<2` or `scikit-learn==1.3.2` (CO-SPY's own `environment.yml` default) — modern
  `torch`/`opencv` wheels expect `numpy>=2`, and `scikit-learn>=1.5` is required to support it.
- Install `torch`/`torchvision` separately, matched to your local CUDA version — they're
  intentionally left out of `requirements.txt`.

## Usage

```python
from model import CoSpyModel

model = CoSpyModel(train_dataset="sd-v1_4")
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1]
```

Or run a full shard-based evaluation against `data/<dataset_name>/`, writing to
`results/co-spy/<dataset_name>.csv`:

```
models/co-spy/venv/bin/python run.py --model co-spy --dataset flux_faces --model-arg train_dataset=sd-v1_4
```
