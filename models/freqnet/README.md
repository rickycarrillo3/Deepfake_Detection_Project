# models/freqnet/

Wrapper around [chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection),
a frequency-domain CNN deepfake detector. Ported from the `FreqNet_setup.ipynb` Colab exploration.

## Setup

```
./models/freqnet/setup.sh
```

This clones `FreqNet-DeepfakeDetection`, creates `models/freqnet/venv/`, and installs both the
repo's own `requirements.txt` and ours. It does **not** install `torch`/`torchvision` — install
those separately, matched to your local CUDA version. The pretrained checkpoint
(`4-classes-freqnet-v2.pth`) ships inside the cloned repo already — no separate download needed.

## Model convention

`freqnet(num_classes=1)` outputs a single logit; `model.py` applies `.sigmoid()` once to get a
fake-probability score directly in `[0, 1]` (higher = more likely fake) — matching
`models/README.md` and `results/README.md`. The notebook's preprocessing (`Resize((256, 256))` +
`ToTensor()`, no normalization) is preserved as-is.

Note: a later cell in the notebook re-applied `sigmoid` to the already-sigmoided score under the
mistaken assumption it was a raw logit (producing a `prob_fake` column). That was a bug in the
notebook's post-hoc analysis, not part of the model's real inference path, so it wasn't ported.

## Usage

```python
from model import FreqNetModel

model = FreqNetModel()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```

Or run a full shard-based evaluation against `data/<dataset_name>/`, writing to
`results/freqnet/<dataset_name>.csv`:

```
models/freqnet/venv/bin/python run.py --model freqnet --dataset flux_faces
```
