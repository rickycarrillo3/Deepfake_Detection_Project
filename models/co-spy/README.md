# models/co-spy/

Wrapper around [Megum1/CO-SPY](https://github.com/Megum1/CO-SPY), a semantic + artifact + fusion
deepfake detector.

## Setup

```bash
cd models/co-spy && bash setup.sh
```

Clones `CO-SPY`, builds `venv/`, installs `requirements.txt` plus `torch`/`torchvision`, and runs
CO-SPY's own `pretrained/download.sh` to fetch the checkpoints. Pass `PYTHON=` to select an
interpreter on a cluster.

Afterwards each `<train_dataset>` should have three files:

```
models/co-spy/CO-SPY/pretrained/<train_dataset>/semantic_weights.pth
models/co-spy/CO-SPY/pretrained/<train_dataset>/artifact_weights.pth
models/co-spy/CO-SPY/pretrained/<train_dataset>/fusion_weights.pth
```

> If you fetch them by hand instead, note that `download.sh` unzips into the **current**
> directory — `cd` into `pretrained/` first. Running it as `sh pretrained/download.sh` from the
> CO-SPY root unpacks one level too high, where the loader will not find them.

## Checkpoints and output naming

`<train_dataset>` selects which checkpoint set to load, and is distinct from the `<dataset_name>`
being evaluated from `data/`. It also determines the generation family in the output filename:
`result_stem()` returns `co-spy_GAN` for `progan` and `co-spy_DM` for anything else.

```bash
models/co-spy/venv/bin/python run.py --model co-spy --dataset flux_test \
  --model-arg train_dataset=sd-v1_4
```

```
results/flux_test_co-spy_GAN_predictions.csv    # --model-arg train_dataset=progan
results/flux_test_co-spy_DM_predictions.csv     # --model-arg train_dataset=sd-v1_4
```

The checkpoint name itself never appears — the family is what separates the two. Both are listed
in `VARIANT_ARGS` in the top-level `run.py`, so a sweep (`--all` / `--models co-spy`) runs each
without the flag being spelled out. Schema in [`results/README.md`](../../results/README.md).

Single image, in Python:

```python
from model import CoSpyModel

model = CoSpyModel(train_dataset="sd-v1_4")
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1]
```

## Implementation notes

Relevant when modifying this adapter, not when running it.

**The two checkpoints use different CLIP families in their semantic branch.** `progan` uses a
HuggingFace `transformers` `CLIPModel`; the version installed here returns a model-output object
from its forward rather than a raw tensor, and its multimodal signature needs a dummy `input_ids`
alongside `pixel_values`. `model.py` monkey-patches `detector.model.sem.forward` at load time to
paper over both — but **only for `progan`** (see `_patch_sem_forward`). `sd-v1_4` uses `open_clip`
SigLIP, whose upstream forward (`self.clip.encode_image(x)`) already returns a tensor; patching it
to call `self.clip(x)` instead yields open_clip's `(image, text, logit_scale)` tuple, which then
fails in `fc` with `linear(): argument 'input' must be Tensor, not tuple`. So it is left alone.

**Checkpoint lookup is CWD-relative upstream.** With `pretrain=True`, CO-SPY's `Detector` ignores
its `ckpt` argument and loads from the *relative* path
`pretrained/<train_dataset>/<branch>_weights.pth`, which assumes `CO-SPY/` is the working
directory. `run.py` imports `model.py` from the repo root, so `load()` constructs the `Detector`
from inside `COSPY_DIR` and restores the CWD afterwards. `pretrain=False` is not an alternative —
it expects `<branch>/best_model.pth`, which is not how the pretrained checkpoints are named.

**Do not pin `numpy<2` or `scikit-learn==1.3.2`** (CO-SPY's own `environment.yml` defaults).
Modern `torch`/`opencv` wheels expect `numpy>=2`, and `scikit-learn>=1.5` is required to support
it. `torch`/`torchvision` are deliberately absent from `requirements.txt` and installed by
`setup.sh` separately, so the CUDA build can be matched to the machine.
