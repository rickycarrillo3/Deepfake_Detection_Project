# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Nine detectors are implemented under `models/`, plus six TrueFake baselines under `baselines/`,
all driven by the shared top-level `run.py`. Each subdirectory's README documents the contract
for code that lives there — read it before implementing anything in that directory.

## Purpose

Benchmarks pretrained deepfake-detection models against shards of the TrueFake dataset, producing
per-model results CSVs for later correlation analysis (used to pick low-correlation models for
ensembling / mixture of experts).

## Architecture

Data flows: `data/` (raw image shards) → `loaders/` (yield `(image_path, label)` pairs) →
`models/<model_name>/` (score each image) →
`results/<dataset_name>_<model_name>_<generation_family>_predictions.csv`.

- **`run.py`** (repo root) — the single runner for every model. There is deliberately **no
  per-model `run.py`**: they were near-identical copies that drifted apart and had to be fixed
  one by one. Models are listed in its `MODELS` registry; adding a detector means writing
  `model.py` and adding one line there.

  Models keep isolated venvs, so invoke it with the venv of the model being run:

  ```
  models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test
  ```

  Flags: `--limit N` (quick test), `--force-label 0|1` (e.g. a fake-only dataset), and
  `--model-arg key=value` (repeatable), which is forwarded to that model's `load()` — this is how
  model-specific config travels (e.g. `--model-arg train_dataset=sd-v1_4` for co-spy) without the
  runner needing to know what it means. `--list` prints the registry.

  Sweep flags run a group of models against one dataset, re-execing each under its own venv and
  cleaning up its clone/venv afterwards: `--all` (everything in `MODELS`), `--baselines` (the six
  `tf-*` entries, derived from `MODELS` by directory), and `--models a,b,c` (an arbitrary named
  subset; repeatable). They are alternatives — pass at most one. All three go through
  `run_all(only=...)`, so cleanup of *shared* clones and venvs is limited to the models that
  actually ran and a subset sweep never deletes a directory it never touched. `--models` names are
  validated up front, so a typo fails before the sweep starts rather than hours into it.

  There is **no** exception: every model, `aeroblade` included, goes through this runner. A model
  whose output does not fit the default CSV schema says so with optional module-level names in its
  `model.py`, read by `run.py` via `getattr` defaults: `SCORE_COLUMN`, `SCORE_PRECISION`,
  `PREDICTED_LABEL`, `SUPPORTS_RESIZE`, and `extra_columns(image_path)`. Only `aeroblade` uses
  them (see below and `models/README.md`); they are for scores that are genuinely not
  fake-probabilities, not a general escape hatch.

- **`data/<dataset_name>/shard_01../`** — git-ignored image shards, 1000 PNGs each, uploaded
  incrementally. The shard count varies by dataset (FFHQ has 20, ~20,000 images), and a dataset
  folder may have only some shards present at any time, so nothing may hardcode a count —
  `loaders/` discovers them by the `shard_` prefix. Ground-truth label is derived
  from the folder name: contains `ffhq` (case-insensitive) → label `0` (real); anything else →
  label `1` (fake).

- **`loaders/`** — shared dataset walking code. Must yield `(image_path, label)` lazily (never
  load all images into memory — runs happen at GPU-cluster scale), and must tolerate partial
  shard uploads by only iterating what currently exists on disk. Yielded `image_path` must be a
  full/absolute path (needed to open the file). The dataset-relative form written to `results/`
  CSVs (e.g. `shard_01/00042.png`) is derived from it via `rel_to_dataset()`.

- **`models/<model_name>/`** — one subdirectory per pretrained detector, each with its own isolated
  environment (`requirements.txt` or `environment.yml`, `venv/`/`.venv/`, `weights/`) since
  different vendored model repos commonly pin conflicting torch/tf versions. There is intentionally
  no shared top-level dependency file. Each model's `model.py` must expose, at **module level**
  (this is what `run.py` calls — a class alone is not enough):
  - `load(**kwargs)` — load weights and prepare the model for inference
  - `predict(image_path) -> score` — return a float score in `[0, 1]`

  Implementing the model as a class is fine; add a thin module-level shim delegating to it (see
  the bottom of `models/freqnet/model.py`). `model.py` must resolve its weights and vendored code
  relative to `__file__`, never the CWD, since `run.py` imports it from the repo root.

  Note `models/DMimageDetection/` holds *two* detectors in subdirectories
  (`DMimageDetection_DM`, `DMimageDetection_GAN` — named for the upstream repo plus the
  generation family each checkpoint was trained on) sharing one venv and weights download, so
  model directories and detectors are not 1:1. The directory name follows the upstream repo,
  grip-unina/DMimageDetection (Corvi et al., ICASSP 2023); the checkpoints it ships are still
  keyed upstream as `Grag2021_progan` / `Grag2021_latent`, which each `model.py` keeps in its
  `MODEL_NAME` — do not rename those. Distinct from `models/grag2021/`, which wraps the
  *different* grip-unina/GANimageDetection repo (Gragnaniello et al., ICME 2021); same lab and
  overlapping co-authors, different paper and different weights.

- **`baselines/<Detector>/`** — the six detectors benchmarked by the TrueFake paper itself
  (Dell'Anna et al., IJCNN 2025), using checkpoints supplied by its authors. Same `model.py`
  contract and same per-directory venv isolation as `models/`, so `run.py` drives them
  identically; registered under `tf-` prefixed names (`tf-npr`, `tf-r50e2p`, …). They sit in
  their own tree because they are one coherent set — all six were retrained by those authors on
  the *same* split — and that shared provenance is what makes their columns comparable to each
  other. It also makes them **not** interchangeable with same-architecture checkpoints under
  `models/`: `baselines/R50_nodown` and `models/DMimageDetection/` are the same network with
  different weights, hence the prefix on every registry name and output CSV.

  Two of them (`R50_TF`, `MISLNet`) vendor no upstream clone — their architectures are
  transcribed into `baselines/tf_networks.py`. `run.py`'s `NO_CLONE_MODELS` set records this so
  `--all` does not read a deliberately absent clone as a broken setup. See `baselines/README.md`,
  which also documents which parts of MISLNet are reconstructed rather than sourced.

- **`results/<dataset_name>_<model_name>_<generation_family>_predictions.csv`** — all results
  live in one flat `results/` directory; the filename carries the dataset, model, and the
  generation family the detector was trained on (`GAN`/`DM`). The `<model>_<family>` middle comes
  from each model.py's optional module-level `result_stem()` (see `run.py`); models without it fall
  back to just `<model_name>` (no family). For checkpoint-varying detectors the stem also encodes
  the checkpoint so runs don't collide: co-spy's family follows `train_dataset` (progan→GAN,
  else→DM), aide's follows `checkpoint` (progan→GAN, sd14→DM, genimage kept literal), and rine
  (always trained on latent-diffusion data, so family `DM`) puts its `ncls` in the model slot —
  `rineldm`, `rine1`, `rine2`, `rine4`. `aeroblade` gets no family suffix
  (`<dataset_name>_aeroblade_predictions.csv`) — training-free, so there is no family it was
  trained on. Output schema is `filepath` (string,
  dataset-relative, e.g. `shard_01/00042.png`; unambiguous alongside the CSV's own
  `<dataset_name>` in its name), `score` (float `[0, 1]`, predicted fake-probability),
  `Predicted label` (int `0`/`1`: `score` thresholded at `metrics.THRESHOLD`), and `resize`
  (string: `"<N>x<N>"` when that run passed `--resize N`, else `"none"`). Written by `run.py`, so
  the schema is uniform by construction. The **ground-truth** label (0 for datasets whose name
  contains `ffhq`, 1 otherwise) is deliberately *not* a CSV column: it is recoverable from the
  dataset name in the filename and is used only to score accuracy on the console.

  `aeroblade` is the one model that departs from that schema, via the hooks listed above. It is
  training-free and returns the LPIPS reconstruction error of the best-matching autoencoder — a
  raw *negative* distance, larger meaning easier to reconstruct, with no real/fake cutoff defined
  anywhere in the paper (it only ever ranks them). So it writes `filepath`, `reconstruction score`
  (that raw distance, **not** in `[0, 1]`), one raw-distance column per autoencoder
  (`stable-diffusion-v1-1`, `kandinsky-2-1` — kept for the correlation analysis), and `resize`.
  No `Predicted label` column and no console accuracy: thresholding an uncalibrated distance at
  `0.5` would invent a cutoff the method does not have. Do not "restore" either one.

  `--resize N` standardizes the input for a fair cross-model comparison: `run.py` feeds every
  model an identical N×N (bicubic) copy of each image *before* that model's own preprocessing,
  rather than each model scoring at native resolution with only its own built-in resize. The
  `resize` column records which regime produced each CSV so runs are not silently mixed. The
  resize is done outside the timed span (it is runner bookkeeping, not model inference — see the
  timing note below). `aeroblade` rejects `--resize` (`SUPPORTS_RESIZE = False`): the resize path
  rewrites every image to one reused temp filename, which its per-directory score cache would
  read as a single one-image directory.

  **Timing goes to `results/run_log.csv`, never into a predictions CSV, and there is still no
  `_summary.csv`.** Every run appends one row — `timestamp`, `model` (the `MODELS` registry name),
  `dataset`, `n_images`, `predict_seconds`, `wall_seconds` — via `metrics.append_run_log()`, called
  from the one shared runner, so every model's runs land in the same log. A sweep
  appends one row per model, since each model runs as its own `run.py --model ...` subprocess.
  Images that fail to score are skipped and counted to the console; they are excluded from
  `n_images` and from every CSV.

  `predict_seconds` is the model's `predict()` calls summed alone; `wall_seconds` is the whole
  invocation, weight loading included. Consolidating the runner is what made the former worth
  recording: one piece of code now times that span identically for every model, where previously
  each model timed itself or not at all. It is still only comparable between rows that shared
  hardware, `--resize` setting and dataset — none of which the log records — so do not read the
  file as a cross-model benchmark table without checking that. Keep durations out of the
  predictions CSVs: their schema is strictly per-image, and the separate file is what stops a
  run-level number from being read as a per-image column.

  At the end of a run, accuracy (thresholded at `0.5`) and inference time are printed **to the
  console** — see `metrics.py`, called from the one place so the threshold and formatting cannot
  drift apart (accuracy is skipped for models whose score is not a fake-probability, i.e.
  `aeroblade`). The reported time is the model's prediction calls summed on their own: each
  `predict()` is timed individually and added up, so it excludes one-off weight loading *and* the
  runner's own work —
  walking the dataset, building rows, writing the CSV. Keep any new timing inside that span only;
  a timer wrapped around the scoring loop silently bills the runner's bookkeeping to the model —
  that wider span is `wall_seconds`, which the run log already records separately.

  Results contents are **git-ignored** (only `.gitkeep` + `README.md` are tracked) — everything
  produced so far has been throwaway test output. Revisit this once real benchmark runs land.
  `run_log.csv` is ignored with them: it records what one machine has run, and an append-only file
  committed from several machines would conflict constantly.

## Conventions to preserve when implementing

- `data/`, `results/` contents, model weight files (`*.pth`, `*.pt`, `*.onnx`, `*.h5`, `*.ckpt`,
  `*.safetensors`), vendored upstream clones, and per-model `venv/`/`.venv/`/`env/` directories are
  git-ignored — do not try to commit these.
- Keep model environments isolated per-directory; do not introduce a shared top-level
  requirements file that would force version alignment across models. `run.py` depends only on
  `pandas` + `tqdm`, which every model venv already has, so it runs under each model's interpreter
  without forcing any version alignment.
- Do not add a per-model `run.py`. Add the model to the `MODELS` registry in the top-level `run.py`
  instead.
- `setup.sh` builds its venv with an overridable interpreter (`PYTHON="${PYTHON:-python3}"`) so a
  cluster's module-loaded Python can be used: `module load python/3.12.12 && PYTHON=python3.12
  ./setup.sh`. Do not hardcode a specific interpreter.
