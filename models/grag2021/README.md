# models/grag2021/

Wrapper around [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection)
(Gragnaniello et al., ICME 2021), a ResNet50-nodown GAN-image detector.

> Not to be confused with [`models/DMimageDetection/`](../DMimageDetection/README.md): same lab and
> overlapping co-authors, but a different repository (grip-unina/DMimageDetection), a different
> paper (Corvi et al., ICASSP 2023) and different weights.

## Setup

```bash
cd models/grag2021 && bash setup.sh
```

Clones `GANimageDetection`, builds `venv/`, installs `requirements.txt` plus
`torch`/`torchvision`, and downloads the `stylegan2` checkpoint into the vendored repository's
`weights/`. Pass `PYTHON=` to select an interpreter on a cluster.

## Score convention — the sign matters

The network outputs a single logit per image and no sigmoid head. **A positive logit means
fake-like.** `model.py` converts to this project's convention with `score = sigmoid(logit)`, so
`score` is a float in `[0, 1]` where higher means more likely fake.

Upstream never states that sign anywhere: `main.py` writes the raw logit to CSV and applies
neither a sigmoid nor a threshold, so no 0.5 exists in the repository or the paper. The natural
decision boundary is `logit = 0`, which is exactly `score = 0.5` once the sigmoid is applied with
the correct sign.

The sign was therefore established empirically, against the 12 ground-truth-labelled images
upstream ships in `example_images/` (`real_*` / `fake_*`), scored with the `stylegan2` checkpoint:

| class | mean logit |
|---|---:|
| `real_*` | **−17.10** |
| `fake_*` | **+14.41** |

Getting this backwards is silent — it inverts every score without erroring — so re-check it if you
swap in a different checkpoint.

## Preprocessing

Images are resized to 256×256 (bilinear) before ImageNet mean/std normalization. This is a
deliberate departure from upstream, which runs at native resolution one image at a time; the
fixed resize trades a small amount of accuracy for much faster batched inference. The runner's
`--resize` applies on top; see [`ResizingRemarks.md`](../../ResizingRemarks.md).

## Running

```bash
models/grag2021/venv/bin/python run.py --model grag2021 --dataset flux_test
```

Trained on GAN-generated images, so its `result_stem()` is `grag2021_GAN` and the run writes
`results/flux_test_grag2021_GAN_predictions.csv` — schema in
[`results/README.md`](../../results/README.md).

Single image, in Python:

```python
from model import Grag2021Model

model = Grag2021Model()
model.load()
score = model.predict("/path/to/image.png")  # float in [0, 1], fake-probability
```
