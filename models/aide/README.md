# models/aide/

Wrapper around [`shilinyan99/AIDE`](https://github.com/shilinyan99/AIDE), ported from the uploaded
`aidecolab.pdf` Colab workflow.

This implementation follows the repository contract:

- `load(**kwargs)` loads the model/checkpoint and prepares inference.
- `predict(image_path) -> score` returns a fake-probability score in `[0, 1]`.
- `result_stem()` names the output file after the checkpoint's generation family (below).

The shared top-level `run.py` writes the standard schema documented in
[`results/README.md`](../../results/README.md) — `filepath`, `score`, `Predicted label`, `resize`.
There is **no `label` column**: ground truth is recoverable from the dataset name (contains `ffhq`
→ real, else fake) and is used only for the console accuracy line. `--force-label 0|1` overrides
that rule for a run whose folder name doesn't follow it.

## Setup

```bash
cd models/aide && bash setup.sh
```

`setup.sh` clones upstream to `AIDE/`, builds `venv/` with an overridable interpreter
(`PYTHON=python3.12 ./setup.sh` on a cluster), installs `requirements.txt` plus torch/torchvision,
and pulls the pretrained checkpoints from the AIDE authors' Google Drive folder into
`AIDE/checkpoints/` with `gdown`. Skip that last step with
`SKIP_CHECKPOINT_DOWNLOAD=1 ./setup.sh`; if Drive rate-limits the download, the script prints the
`gdown --folder ...` command to re-run by hand.

## Checkpoints

AIDE ships three detector checkpoints, trained on different fake-image sources. Pick one with
`--model-arg checkpoint=<name>`:

| `checkpoint` | file | trained on | result stem |
|---|---|---|---|
| `genimage` (default) | `GenImage_train.pth` | GenImage (mixed generators) | `aide_genimage` |
| `progan` | `progan_train.pth` | ProGAN | `aide_GAN` |
| `sd14` | `sd14_train.pth` | Stable Diffusion 1.4 | `aide_DM` |

`genimage` keeps its own literal label rather than being forced into `GAN`/`DM`, because a mixed
generator set is neither. The differing stems also keep the three checkpoints' runs in separate
CSVs; all three are listed in `VARIANT_ARGS` in the top-level `run.py`, so a sweep (`--all` /
`--models aide`) runs each of them.

To use a checkpoint from elsewhere, give a full path — it overrides `checkpoint=` entirely:

```bash
models/aide/venv/bin/python run.py --model aide --dataset flux_faces \
  --model-arg checkpoint_path=/absolute/path/to/your_checkpoint.pth
```

`AIDE_WEIGHTS_PATH=/absolute/path/to/checkpoint.pth` in the environment does the same thing.

## Model convention

AIDE outputs two logits:

```text
class 0 = real-like
class 1 = fake / AI-generated-like
```

This wrapper returns:

```python
score = softmax([logit_real, logit_fake])[1]
```

so higher `score` means "more fake-like."

## Usage

Quick single-image test:

```python
from model import AIDEModel

model = AIDEModel()
model.load()
score = model.predict("/path/to/image.png")
print(score)
```

Full dataset run:

```bash
models/aide/venv/bin/python run.py --model aide --dataset flux_faces
```

Quick smoke test:

```bash
models/aide/venv/bin/python run.py --model aide --dataset flux_faces --limit 10
```

With the default `genimage` checkpoint that writes:

```text
results/flux_faces_aide_genimage_predictions.csv
```

with these columns:

```text
filepath,score,Predicted label,resize
```

No per-run timing CSV is written — the run's duration is appended to `results/run_log.csv`
instead. See [`results/README.md`](../../results/README.md) for why.

## Notes

AIDE preprocessing is heavier than many other detectors because each image is converted into five
input tensors:

1. DCT/noise low-rank patch
2. DCT/noise high-rank patch
3. second low-rank patch
4. second high-rank patch
5. resized/normalized image tensor

This matches the uploaded Colab workflow and is why AIDE may run slower than simpler resize-only
models.
