# models/aide/

Wrapper around [`shilinyan99/AIDE`](https://github.com/shilinyan99/AIDE).

## Setup

```bash
cd models/aide && bash setup.sh
```

Clones upstream to `AIDE/`, builds `venv/`, installs `requirements.txt` plus
`torch`/`torchvision`, and pulls the pretrained checkpoints from the AIDE authors' Google Drive
folder into `AIDE/checkpoints/` with `gdown`. Pass `PYTHON=` to select an interpreter on a cluster
(`PYTHON=python3.12 bash setup.sh`).

`SKIP_CHECKPOINT_DOWNLOAD=1` skips the download. If Drive rate-limits it — which is a Google
limit, not a problem with this repo — the script prints the `gdown --folder ...` command to re-run
by hand.

## Checkpoints

AIDE ships three detector checkpoints, trained on different fake-image sources. Pick one with
`--model-arg checkpoint=<name>`:

| `checkpoint` | file | trained on | result stem |
|---|---|---|---|
| `genimage` (default) | `GenImage_train.pth` | GenImage (mixed generators) | `aide_genimage` |
| `progan` | `progan_train.pth` | ProGAN | `aide_GAN` |
| `sd14` | `sd14_train.pth` | Stable Diffusion 1.4 | `aide_DM` |

`genimage` keeps its own literal label rather than being forced into `GAN`/`DM`, because a mixed
generator set is neither. The differing stems keep the three checkpoints' runs in separate CSVs;
all three are listed in `VARIANT_ARGS` in the top-level `run.py`, so a sweep (`--all` /
`--models aide`) runs each of them.

A checkpoint from elsewhere can be given as a full path, which overrides `checkpoint=` entirely:

```bash
models/aide/venv/bin/python run.py --model aide --dataset flux_test \
  --model-arg checkpoint_path=/absolute/path/to/your_checkpoint.pth
```

`AIDE_WEIGHTS_PATH=/absolute/path/to/checkpoint.pth` in the environment does the same thing.

## Score convention

AIDE outputs two logits — class 0 real-like, class 1 fake-like. This wrapper returns
`softmax([logit_real, logit_fake])[1]`, so a higher `score` means more fake-like, in `[0, 1]` as
[`results/README.md`](../../results/README.md) specifies.

## Running

```bash
models/aide/venv/bin/python run.py --model aide --dataset flux_test --limit 10   # smoke test
models/aide/venv/bin/python run.py --model aide --dataset flux_test
```

With the default `genimage` checkpoint that writes
`results/flux_test_aide_genimage_predictions.csv`.

Single image, in Python:

```python
from model import AIDEModel

model = AIDEModel()
model.load()
score = model.predict("/path/to/image.png")
```

## Preprocessing cost

AIDE's preprocessing is heavier than most detectors here: each image is converted into **five**
input tensors — a DCT/noise low-rank patch, a DCT/noise high-rank patch, a second low-rank patch,
a second high-rank patch, and a resized/normalized image tensor. Expect it to be slower than the
resize-only detectors; see [`ResizingRemarks.md`](../../ResizingRemarks.md).
