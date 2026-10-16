# models/

The nine detectors evaluated with their **original authors' checkpoints**, one subdirectory each.

The six detectors retrained by the TrueFake paper's authors live in
[`baselines/`](../baselines/README.md) instead, `tf-` prefixed — same contract, different
provenance, and deliberately not interchangeable with these.

## The nine detectors

| registry name | dir | detector | upstream |
|---|---|---|---|
| `cnndetection` | `cnndetection/` | CNNDetection | Wang et al., CVPR 2020 — [PeterWang512/CNNDetection](https://github.com/PeterWang512/CNNDetection) |
| `grag2021` | `grag2021/` | GAN image detection (ResNet-50 no-down) | Gragnaniello et al., ICME 2021 — [grip-unina/GANimageDetection](https://github.com/grip-unina/GANimageDetection) |
| `DMimageDetection_GAN` | `DMimageDetection/DMimageDetection_GAN/` | ProGAN checkpoint | Corvi et al., ICASSP 2023 — [grip-unina/DMimageDetection](https://github.com/grip-unina/DMimageDetection) |
| `DMimageDetection_DM` | `DMimageDetection/DMimageDetection_DM/` | latent-diffusion checkpoint | same repo |
| `freqnet` | `freqnet/` | FreqNet — frequency-domain CNN | [chuangchuangtan/FreqNet-DeepfakeDetection](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) |
| `rine` | `rine/` | RINE — intermediate CLIP encoder blocks | [mever-team/rine](https://github.com/mever-team/rine) |
| `aide` | `aide/` | AIDE | [shilinyan99/AIDE](https://github.com/shilinyan99/AIDE) |
| `co-spy` | `co-spy/` | Co-Spy — semantic + artifact fusion | [Megum1/CO-SPY](https://github.com/Megum1/CO-SPY) |
| `aeroblade` | `aeroblade/` | AEROBLADE — training-free reconstruction error | CVPR 2024 — [jonasricker/aeroblade](https://github.com/jonasricker/aeroblade) |

> `grag2021` and `DMimageDetection` are easy to conflate: same lab and overlapping co-authors, but
> different repositories, different papers and different weights. The upstream checkpoint keys
> `Grag2021_progan` / `Grag2021_latent` are retained verbatim in each `DMimageDetection`
> `model.py`, which is why that older name appears inside the newer repo's code.

Directories are **not** one-to-one with detectors: `models/DMimageDetection/` holds two, in
subdirectories sharing a single venv and weights download.

## Setup and running

Each detector owns an isolated environment — upstream repositories pin conflicting torch/TF
versions, so there is intentionally no shared top-level requirements file. Set one up, then invoke
the shared top-level runner with **that detector's interpreter**:

```bash
cd models/freqnet && bash setup.sh && cd ../..
models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test --limit 10
```

What each `setup.sh` has to fetch, and the interpreter it was tested against:

| detector | checkpoint source | Python |
|---|---|---|
| `cnndetection` | upstream's `weights/download_weights.sh`, run automatically | 3.12 |
| `grag2021` | direct download from `grip.unina.it` | 3.12 |
| `DMimageDetection_*` | Google Drive (`gdown`), one shared `weights.zip` for both | 3.12 |
| `freqnet` | ships inside the cloned repository — no separate download | 3.12 |
| `rine` | ships inside the cloned repository (`ckpt/`, symlinked to the repo root) | **3.9** |
| `aide` | Google Drive folder (`gdown`) | 3.12 |
| `co-spy` | upstream's `pretrained/download.sh`, run automatically | 3.12 |
| `aeroblade` | HuggingFace autoencoders (~430 MB) | **3.10** |

Pass the interpreter explicitly where the default `python3` is wrong for a detector or too old:

```bash
PYTHON=python3.9  bash models/rine/setup.sh        # upstream pins torch against 3.9
PYTHON=python3.10 bash models/aeroblade/setup.sh   # torch==2.1.2 has no wheels for 3.12+
```

Two fetches are worth anticipating. The Google Drive downloads (`aide`, `DMimageDetection`) can
fail on Google's rate limits rather than on anything in this repo; `models/aide/setup.sh` prints
the manual `gdown` command to retry with. And **set `HF_HOME` to scratch space before running
`aeroblade`**, not a quota'd home directory.

## Variants

Three detectors ship more than one meaningfully different checkpoint. A sweep (`--all`,
`--baselines`, `--models`) expands these automatically, so the full sweep is 21 runs rather than
15:

| detector | variants | selected with |
|---|---|---|
| `co-spy` | ProGAN, SD-v1.4 | `--model-arg train_dataset=progan\|sd-v1_4` |
| `aide` | GenImage, ProGAN, SD-1.4 | `--model-arg checkpoint=genimage\|progan\|sd14` |
| `rine` | LDM, and 1-, 2-, 4-class ProGAN | `--model-arg ncls=ldm\|1\|2\|4` |

Each variant writes its own CSV — the distinguishing value is folded into the filename, so runs
never overwrite each other. See [`results/README.md`](../results/README.md) for the full table of
output names.

## Layout

```
models/<model_name>/
├── model.py              # load() / predict() — the contract run.py calls
├── setup.sh              # clones upstream, builds the venv, fetches weights
├── requirements.txt      # or environment.yml — this detector's own dependencies
├── venv/                 # isolated environment (git-ignored)
├── weights/              # checkpoints (git-ignored)
└── <upstream clone>/     # vendored copy of the detector's repository (git-ignored)
```

There is no per-detector `run.py`; the top-level `run.py` drives all of them. Several directories
carry their own `README.md` with detector-specific notes — `models/aeroblade/README.md` in
particular documents its disk footprint and its departures from the standard output schema.

## `aeroblade` is the one departure from the output contract

Every other detector here returns a fake-probability in `[0, 1]`, thresholded at
`metrics.THRESHOLD` into a `Predicted label` column. AEROBLADE is training-free and returns a raw
LPIPS reconstruction distance (negative; larger means easier to reconstruct) for which the paper
defines no real/fake cutoff — it only ever ranks images. Its CSV therefore carries a
`reconstruction score` and one raw-distance column per autoencoder, with no predicted label and no
reported accuracy, and it declines `--resize`. It is consequently out of scope for `ensemble.py`
and `failure_correlation.py`. See [`aeroblade/README.md`](aeroblade/README.md).

## Adding a detector

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the `model.py` contract, the `setup.sh` shape, and
how to register a new detector.
