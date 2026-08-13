# models/grag2021/

Wrapper around [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection)
(Grag2021), a ResNet50-nodown GAN-image detector. Ported from the `Grag2021_test.ipynb` Colab
exploration.

## Setup

```
./models/grag2021/setup.sh
```

This clones `GANimageDetection`, creates `models/grag2021/venv/`, installs `requirements.txt`, and
downloads the pretrained checkpoint into the vendored repo's `weights/` folder. It does **not**
install `torch`/`torchvision` — install those separately, matched to your local CUDA version.

## Model convention

The network outputs a single logit per image (no sigmoid head). **Positive logit means fake-like.**
`model.py` converts this to the project's fake-probability-score convention via
`score = sigmoid(logit)`, so `score` stays a float in `[0, 1]` where higher = more likely fake,
consistent with `models/README.md` and `results/README.md`.

Upstream never states this sign, which is how it came to be inverted here: `main.py` writes the raw
logit to CSV and applies no sigmoid and no threshold, so there is no 0.5 in the repo or the paper —
the natural decision boundary is `logit = 0`, which is exactly `score = 0.5` once the sigmoid is
applied with the correct sign. The convention was established empirically against the 12
ground-truth-labeled images the upstream repo ships in `example_images/` (`real_*` / `fake_*`),
scored with the `stylegan2` checkpoint:

| class | mean logit |
|-------|-----------:|
| `real_*` | **-17.10** |
| `fake_*` | **+14.41** |

Before this was corrected, `model.py` used `sigmoid(-logit)`, which inverted every score: real
images came out at ~0.99 and fakes at ~0.01. If you have results CSVs produced by the old code,
they are wrong and should be regenerated, not patched — a saved `score` can be recovered as
`1 - score`, but re-running is safer than trusting which files predate the fix.

Images are resized to 256x256 (bilinear) before normalization with ImageNet mean/std — this
matches the notebook's `FORCE_RESIZE=True` fast path, which trades a small amount of accuracy for
much faster batched inference over the original (which runs at native resolution, one image at a
time).

## Usage

```python
from model import Grag2021Model

model = Grag2021Model()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```

Or run a full shard-based evaluation against `data/<dataset_name>/` through the shared runner:

```
models/grag2021/venv/bin/python run.py --model grag2021 --dataset flux_faces
```

Grag2021 was trained on GAN-generated images, so its `result_stem()` is `grag2021_GAN` and the run
writes `results/flux_faces_grag2021_GAN_predictions.csv` — schema in
[`results/README.md`](../../results/README.md).

Not to be confused with [`models/DMimageDetection/`](../DMimageDetection/README.md): same lab and
overlapping co-authors, but a different repo (grip-unina/DMimageDetection), a different paper
(Corvi et al., ICASSP 2023) and different weights.
