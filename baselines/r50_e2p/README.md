# r50_e2p — TrueFake's R50-E2P

*ResNet50 as Feature Encoder to Prototypes*, the detector proposed in the TrueFake paper.

- Upstream: <https://github.com/MMLab-unitn/TrueFake-IJCNN25> (`./detector/`)
- Paper: <https://arxiv.org/abs/2504.20658>

## Architecture

A `nodown` ResNet50 (Gragnaniello et al.'s modification replacing the stride-2 7×7 first conv with
a stride-1 one, so no early downsampling discards high-frequency forensic cues) with its
**ImageNet weights frozen**, used purely as a feature encoder. The only trained parts are a
2048→128 fully-connected head and a `ScoresLayer` of learned Gaussian prototypes on top of it.

Upstream expresses this as flags on one `ImageClassifier` rather than as a distinct architecture —
`--arch nodown --freeze --prototype` — so `model.py` reconstructs those flags in a settings shim
and calls upstream's class directly rather than reimplementing it.

## Checkpoint

One checkpoint is published, trained on data keys
`gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre` — StyleGAN2 and StableDiffusion XL fakes against
FFHQ and FORLAB reals, all non-shared (pre-social). Because it saw both a GAN and a diffusion
generator, `result_stem()` reports family **`GANDM`** rather than `GAN` or `DM`:

```
results/baseline_<dataset>_r50_e2p_GANDM_predictions.csv
```

Upstream cautions that these weights were trained on a deliberately limited subset in order to
replicate the paper's numbers, and should be retrained on something broader for any other purpose.

## Scores

Upstream trains with `BCEWithLogitsLoss` against `Fake = 1` and thresholds the raw output at `0`.
The network output is therefore a **logit**, not a probability. `predict()` applies a sigmoid,
which satisfies this repo's `[0, 1]` contract and maps upstream's `0` cutoff exactly onto
`metrics.THRESHOLD` (0.5) — so a prediction thresholded here matches one thresholded upstream.

## Preprocessing and image size

Upstream's **test** transform applies no resize and no crop (`ToImage` → `float32`/scale →
ImageNet normalize); the network is fully convolutional and scores at native resolution. `model.py`
reproduces that exactly and deliberately adds no resize of its own.

Two consequences worth knowing before a run:

- **Memory.** With `nodown`'s stride-1 first conv, activations are 4× larger in each spatial
  dimension than a standard ResNet50's. A native-resolution 1024×1024 TrueFake image is heavy;
  the paper ran this on a 24 GB RTX 4090 at batch size 8. Here it is one image at a time, but a
  CPU-only smoke test will still be slow.
- Use `run.py --resize N` when a standardized input size is wanted. That keeps the resize regime
  recorded in the CSV's `resize` column instead of baking it silently into this wrapper — but note
  that resizing away from native resolution is a departure from how the paper evaluated, so
  `--resize` runs are not directly comparable to the paper's published numbers.

The FORLAB-specific `CenterCrop`s in upstream's `transform_end` are conditioned on `realFORLAB:*`
appearing in that run's data keys. They do not apply to this repo's datasets and are omitted.

## Setup

```bash
./setup.sh                      # clone + venv + fetch weights/best.pt from Google Drive
PYTHON=python3.12 ./setup.sh    # on a cluster, after `module load python/3.12.12`
```

The checkpoint is a Google Drive download, fetched with `gdown` (which handles Drive's large-file
confirm interstitial). If it fails — Drive occasionally rate-limits — grab it manually from the
link in the upstream README's "Use pretrained network" section and save it as
`baselines/r50_e2p/weights/best.pt`.

## Run

```bash
baselines/r50_e2p/venv/bin/python run.py --model r50_e2p --dataset ffhq_test
```

Optional `--model-arg` passthroughs, all defaulting to the published checkpoint's configuration:
`checkpoint=<path>`, `arch=nodown|baseline`, `prototype=true|false`, `freeze=true|false`,
`num_centers=<int>`. They exist so a differently-configured checkpoint can be loaded without
editing `model.py`; the defaults are the ones that match `weights/best.pt`, and changing `arch` or
`prototype` will fail to load that checkpoint.
