# [TODO: PAPER TITLE]

Code and benchmarking harness for:

> **[TODO: PAPER TITLE]**
> [TODO: AUTHOR LIST]
> [TODO: VENUE, YEAR] · [TODO: arXiv / DOI link]

This repository benchmarks **fourteen pretrained deepfake detectors** against the
[TrueFake](https://arxiv.org/abs/2504.20658) dataset under a single, uniform evaluation harness,
and emits per-image scores as CSVs. The scores are then analysed for **failure correlation**
between detectors — the question being which detectors fail on *different* images, and are
therefore worth combining into an ensemble or mixture of experts.

A fifteenth detector, AEROBLADE, is implemented and driven by the same runner but **was not
included in the reported benchmark**. It is kept here because it is training-free and returns a
reconstruction distance rather than a fake-probability, which makes it a useful reference point;
see [`models/aeroblade/`](models/aeroblade/README.md).

The harness exists because per-detector evaluation code is not comparable across detectors: each
upstream repository preprocesses images differently, thresholds differently, and times itself
differently. Here every detector is driven by one runner (`run.py`), scored at one threshold
(`metrics.THRESHOLD`), and timed over one identically-measured span, so the resulting columns can
be placed side by side.

## Detectors under test

Eight of the benchmarked detectors use their **original authors' checkpoints**
([`models/`](models/README.md)), alongside the unbenchmarked AEROBLADE:

| registry name | detector | source |
|---|---|---|
| `cnndetection` | CNNDetection | Wang et al., CVPR 2020 — [PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection) |
| `grag2021` | GAN image detection | Gragnaniello et al., ICME 2021 — [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection) |
| `DMimageDetection_GAN` | ResNet-50 no-down, ProGAN checkpoint | Corvi et al., ICASSP 2023 — [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) |
| `DMimageDetection_DM` | same repo, latent-diffusion checkpoint | Corvi et al., ICASSP 2023 |
| `freqnet` | FreqNet (frequency-domain CNN) | [chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) |
| `rine` | RINE (intermediate CLIP encoder blocks) | [mever-team/rine](https://github.com/mever-team/rine) |
| `aide` | AIDE | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) |
| `co-spy` | Co-Spy (semantic + artifact fusion) | [Megum1/CO-SPY](https://github.com/Megum1/CO-SPY) |
| `aeroblade` *(not benchmarked)* | AEROBLADE (training-free reconstruction error) | CVPR 2024 — [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) |

Six are the **TrueFake paper's own baselines** ([`baselines/`](baselines/README.md)), using
checkpoints supplied by that paper's authors and prefixed `tf-`:

| registry name | detector | source |
|---|---|---|
| `tf-r50e2p` | R50-E2P — the TrueFake paper's proposed method | Dell'Anna et al., IJCNN 2025 |
| `tf-r50nd` | R50-ND | Corvi et al., ICASSP 2023 |
| `tf-npr` | NPR | Tan et al., CVPR 2024 |
| `tf-mislnet` | MISLNet (Bayar & Stamm constrained conv) | as used by Vahdati et al., CVPRW 2024 |
| `tf-clipd` | CLIP-D | Cozzolino et al., CVPRW 2024 |
| `tf-p2g` | Prompt2Guard | Laiti et al., 2025 |

The `tf-` prefix is load-bearing rather than cosmetic. All six were **retrained by the TrueFake
authors on one shared split** (fakes from StyleGAN2 + SDXL, reals from FFHQ + FORLAB), which is
what makes their columns comparable to each other — and what makes them *not* interchangeable
with the same architectures under `models/`. `baselines/R50_nodown` and
`models/DMimageDetection/` are the same network with different weights.

Three detectors ship more than one meaningfully different checkpoint, which the runner expands
automatically: `co-spy` (ProGAN / SD-v1.4), `aide` (GenImage / ProGAN / SD-1.4) and `rine` (LDM,
and 1-, 2-, 4-class ProGAN). The fourteen benchmarked detectors therefore produce **20 scored
runs**, and `--all`, which also covers AEROBLADE, produces 21 across all 15 registry entries.

## Requirements

- **Linux or macOS**, with `bash` and `git`.
- **Python 3.12** for most detectors. Three need something older, because their upstream pins
  have no wheels for newer interpreters: `aeroblade` needs **≤ 3.11** (upstream pins
  `torch==2.1.2`; 3.10 is the tested choice), `rine` needs **3.9**, and `tf-p2g` accepts
  **3.8–3.12**. Each `setup.sh` checks its own interpreter and says so before failing.
- **An NVIDIA GPU is strongly recommended but not required.** Every detector selects
  `cuda:0` when available and falls back to CPU otherwise. The pinned wheels target **CUDA 11.8**.
- **Disk.** Each detector builds its own venv and downloads its own weights; budget a few GB per
  detector. `aeroblade` additionally needs ~430 MB of autoencoder weights — point `HF_HOME` at
  scratch space before running it.

There is deliberately **no shared top-level requirements file**: upstream repositories pin
conflicting torch/TF versions, so each detector owns an isolated environment. The runner itself
needs only `pandas` + `tqdm`, which every detector's venv already provides.

## Installation

Each detector is set up independently. `setup.sh` clones the upstream repository, builds the venv
and fetches weights:

```bash
cd models/freqnet && bash setup.sh && cd ../..
```

On a cluster, select the interpreter explicitly rather than relying on the default `python3`:

```bash
module load python/3.12.12
PYTHON=python3.12 bash models/freqnet/setup.sh
```

> **The six `tf-*` baseline checkpoints are not publicly hosted** — they were provided directly by
> the TrueFake authors. Their `setup.sh` scripts build the environment but cannot fetch the
> weights; place the checkpoint in each baseline's directory yourself, then verify it with
> `baselines/NPR/venv/bin/python baselines/verify_checkpoints.py NPR` (one baseline per venv —
> the environments are deliberately split). That check loads the checkpoint with `strict=True`
> and runs one forward pass, because two of the six architectures are transcribed rather than
> imported, and a wrong transcription produces plausible-looking scores rather than an error. See
> [`baselines/README.md`](baselines/README.md).

## Dataset

The benchmark reads image shards from `data/`, which is git-ignored — obtain TrueFake from the
authors' release ([arXiv:2504.20658](https://arxiv.org/abs/2504.20658),
[MMLab-unitn/TrueFake-IJCNN25](https://github.com/MMLab-unitn/TrueFake-IJCNN25)) and arrange it as:

```
data/<dataset_name>/shard_01/<images>
data/<dataset_name>/shard_02/<images>
...
```

Shards hold 1000 images each by convention, in PNG or JPEG. The shard count varies by dataset and
shards need not all be present at once — the loader discovers whatever exists by the `shard_`
prefix, so a partial download is usable. Subdirectories not named `shard_*`, and files with any
other extension, are skipped silently, so a tree arranged differently scores nothing rather than
failing. **Ground truth comes from the directory name**: a `<dataset_name>` containing `ffhq`
(case-insensitive) is real (`0`); anything else is fake (`1`). Each dataset is therefore
single-class by construction, which is why AUROC is computed by pooling a real dataset with a fake
one after the fact rather than within a run.

## Reproducing the benchmark

**1. Score each detector.** Invoke the shared runner with the detector's *own* interpreter:

```bash
models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test
```

Add `--limit 10` to confirm wiring before committing to a full dataset.

**2. Or sweep a group.** Sweeps re-exec each detector under its own venv, so launch them with a
plain interpreter that has `pandas` + `tqdm`:

```bash
python3 run.py --list                                        # the registry
python3 run.py --all        --dataset ffhq_test              # all 15, AEROBLADE included (21 runs)
python3 run.py --baselines  --dataset ffhq_test              # the six tf-* only
python3 run.py --models tf-npr,freqnet --dataset ffhq_test   # a named subset
```

Each sweep deletes a detector's clone and venv after it finishes, to reclaim disk; `--keep-repo` /
`--keep-venv` opt out, and `--dry-run` prints the plan without running or deleting anything. Do
**not** launch a sweep from one of the detectors' own venvs — it would try to delete the venv out
from under the running interpreter. (This is detected and skipped rather than crashed, but that
venv then goes uncleaned.)

**3. Analyse the finished CSVs.** All three scripts read the flat `results/` directory and need
only pandas (+ numpy/matplotlib), so they run under any detector venv or a plain interpreter:

```bash
python3 auroc_report.py                 # pooled real-vs-fake AUROC per detector
python3 ensemble.py --dataset flux_test # do combined scores beat their members?
python3 failure_correlation.py          # failure-correlation matrices and plots
```

| script | what it reports | output |
|---|---|---|
| `auroc_report.py` | pairs each fake dataset's CSV with the ffhq CSV *for the same detector*, pools the scores, reports AUROC + accuracy + per-side rates | console, `results/auroc_summary.csv` |
| `ensemble.py` | combines per-detector scores (`mean`, `logodds`, `hard_or`; `--norm raw/rank/zlogit`) and reports whether the ensemble beats its members | console, optional `--out DIR` |
| `failure_correlation.py` | per-dataset failure correlations and oracle accuracy across detectors | `results/plots/<dataset>/` |

`score_correlation_publish.ipynb` carries the correlation analysis as presented in the paper.

Dataset discovery and the `<dataset>_<stem>_predictions.csv` filename split come from
`ensemble.py`, and ground truth from `loaders.label_for_dataset`, so all three scripts see the
same CSVs and agree on what is real. Datasets are discovered from the directory names under
`data/`, which therefore has to be present even when only finished CSVs are being analysed — both
halves of a results filename contain underscores, so the split cannot be inferred from the
filename alone.

### Controlled-resolution runs

By default each detector sees images at native resolution and applies only its own built-in
resizing — the regime its authors evaluated under. `--resize N` instead feeds **every** detector
an identical N×N bicubic copy *before* its own preprocessing, which is the controlled comparison:

```bash
python3 run.py --all --dataset ffhq_test --resize 256
```

Every CSV records which regime produced it in a `resize` column, and the analysis scripts refuse
to pool across regimes. The resize is excluded from the timed span, being harness bookkeeping
rather than inference. `aeroblade` rejects `--resize` outright. See
[`ResizingRemarks.md`](ResizingRemarks.md) for what each detector does to an image before its
network sees it, and why the native-resolution ones are the slow ones.

## Outputs

### Prediction CSVs

One CSV per run, in a flat directory:

```
results/<dataset_name>_<model_name>_<generation_family>_predictions.csv
```

| column | type | description |
|---|---|---|
| `filepath` | string | dataset-relative, e.g. `shard_01/00042.png` |
| `score` | float `[0, 1]` | predicted fake-probability |
| `Predicted label` | int `0`/`1` | `score` thresholded at `metrics.THRESHOLD` (0.5) |
| `resize` | string | `"<N>x<N>"` when the run passed `--resize N`, else `"none"` |

The ground-truth label is deliberately *not* a column: it is recoverable from the dataset name in
the filename, and keeping one source of truth prevents the two from disagreeing.

The `<model>_<family>` middle of the filename records the generation family the detector was
trained on, and distinguishes checkpoints of the same detector:

| detector | stems |
|---|---|
| freqnet, grag2021, cnndetection | `freqnet_GAN`, `grag2021_GAN`, `cnndetection_GAN` |
| DMimageDetection | `DMimageDetection_GAN`, `DMimageDetection_DM` |
| co-spy | `co-spy_GAN` (ProGAN), `co-spy_DM` (any other checkpoint) |
| aide | `aide_GAN` (ProGAN), `aide_DM` (SD-1.4), `aide_genimage` (a mixed-generator set, kept literal) |
| rine | `rine_DM` (LDM), `rine1_GAN` / `rine2_GAN` / `rine4_GAN` (ProGAN-trained N-class) |
| aeroblade | `aeroblade` — no family; training-free, so there is none |
| the six baselines | `tf-<name>_GANDM`, e.g. `tf-npr_GANDM` — one split spanning **both** families |

For example: `ffhq_test_rine1_GAN_predictions.csv`, `flux_test_tf-npr_GANDM_predictions.csv`.

**`aeroblade` is the one departure from this schema.** It is training-free and returns a raw LPIPS
reconstruction distance (negative; larger means easier to reconstruct) for which the paper defines
no real/fake cutoff — it only ever ranks. Its CSV carries `filepath`, `reconstruction score`, one
raw-distance column per autoencoder (`stable-diffusion-v1-1`, `kandinsky-2-1`) and `resize`, with
no `Predicted label` and no reported accuracy, since thresholding an uncalibrated distance at 0.5
would invent a cutoff the method does not have. It is consequently out of scope for `ensemble.py`
and `failure_correlation.py`, which skip it with a notice.

### Timing

Every run appends one row to `results/run_log.csv` (`timestamp`, `model`, `dataset`, `n_images`,
`predict_seconds`, `wall_seconds`). `predict_seconds` sums the detector's `predict()` calls alone
— excluding weight loading, dataset walking and CSV writing — and is measured by the same code for
every detector, which is what makes it comparable at all. `wall_seconds` is the whole invocation,
and is an operational number for planning cluster time rather than a property of the detector.

Rows are comparable only when hardware, `--resize` regime and dataset all matched. The log records
none of the first two, so it is not a cross-model benchmark table on its own.

## Repository layout

```
run.py                  the one runner for every detector (MODELS registry lives here)
metrics.py              THRESHOLD, accuracy, AUROC, console summary, run-log append
loaders/                shared dataset walking
data/                   image shards (git-ignored)
models/                 nine detectors, original-author checkpoints
baselines/              six TrueFake-paper detectors, that paper's checkpoints
results/                prediction CSVs + run_log.csv (contents git-ignored)
auroc_report.py         pooled real-vs-fake AUROC across finished CSVs
ensemble.py             ensemble experiments over finished CSVs
failure_correlation.py  per-dataset failure-correlation plots and matrices
score_correlation_publish.ipynb   the correlation analysis reported in the paper
```

Data flows `data/` → `loaders/` (yields `(image_path, label)` lazily) → `models/<name>/model.py`
(scores each image) → `results/*.csv`, with `run.py` driving every stage.

Each subdirectory has its own README documenting the contract for the code that lives there.

## Citation

```bibtex
@inproceedings{TODO_CITATION_KEY,
  title     = {TODO: PAPER TITLE},
  author    = {TODO: AUTHOR LIST},
  booktitle = {TODO: VENUE},
  year      = {TODO: YEAR}
}
```

If you use this benchmark, please also cite the TrueFake dataset and the individual detectors you
report — each detector's upstream paper is listed in the tables above, and in
[`models/README.md`](models/README.md) / [`baselines/README.md`](baselines/README.md):

```bibtex
@inproceedings{dellanna2025truefake,
  title     = {TrueFake: A Real World Case Dataset of Last Generation Fake Images
               also Shared on Social Networks},
  author    = {Dell'Anna, Stefano and Montibeller, Andrea and Boato, Giulia},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025}
}
```

## License

This harness is released under the MIT License — see [LICENSE](LICENSE).

The licenses of the vendored upstream detector repositories are **unaffected** by this and
continue to govern their own code and weights. `setup.sh` clones each one from its original
source rather than redistributing it, so check the terms of any detector you build on. The
TrueFake dataset and the `tf-*` checkpoints are the property of the TrueFake authors and are
distributed on their terms, not ours.

## Extending the benchmark

[CONTRIBUTING.md](CONTRIBUTING.md) is a step-by-step guide to adding a detector: the `model.py`
contract, the `setup.sh` shape, registering it, and verifying the result.
