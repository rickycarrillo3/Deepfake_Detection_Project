# baselines/

The six detectors benchmarked by the TrueFake paper — Dell'Anna, Montibeller & Boato, *"TrueFake:
A Real World Case Dataset of Last Generation Fake Images also Shared on Social Networks"*, IJCNN
2025 ([arXiv:2504.20658][paper]) — using checkpoints supplied by that paper's authors.

[paper]: https://arxiv.org/abs/2504.20658
[code]: https://github.com/MMLab-unitn/TrueFake-IJCNN25

## The six

| registry name | dir | detector | architecture from |
|---|---|---|---|
| `tf-r50e2p` | `R50_TF/` | **R50-E2P** — the paper's own proposed method (§IV.6): no-downsample ResNet-50 → 128-d embedding → learned-prototype head | authors' [code][code], `detector/networks.py` |
| `tf-r50nd` | `R50_nodown/` | **R50-ND** — Corvi et al., ICASSP 2023 | vendored grip-unina clone |
| `tf-npr` | `NPR/` | **NPR** — Tan et al., CVPR 2024 | vendored upstream clone |
| `tf-mislnet` | `MISLNet/` | **MISLNet** — Bayar & Stamm constrained conv, as used by Vahdati et al., CVPRW 2024 | **reconstructed from the checkpoint** — see the caveat below |
| `tf-clipd` | `CLIP-D/` | **CLIP-D** — Cozzolino et al., CVPRW 2024 | vendored grip-unina clone |
| `tf-p2g` | `P2G/` | **Prompt2Guard** — Laiti et al., 2025 | vendored upstream clone |

`R50_TF` and `R50_nodown` keep the directory names their checkpoints arrived with; R50-E2P and
R50-ND are the paper's names for them.

## Why these are separate from `models/`

They use the same `model.py` contract, so the top-level `run.py` drives them identically. They sit
in their own tree because they are **one coherent set** rather than six independent detectors: all
six were retrained by the TrueFake authors on the same split — fakes from StyleGAN2 + SDXL, reals
from FFHQ + FORLAB, 70/15/15. That shared provenance is the point of the set. It is what makes
their columns comparable to one another, and what makes them *not* interchangeable with the same
architectures under `models/`.

Concretely, `baselines/R50_nodown` and `models/DMimageDetection/` are the same network with
different weights. Conflating them would silently compare a TrueFake-retrained detector against an
original-author one. Hence the `tf-` prefix on every registry name and every output CSV.

## Checkpoints

> **The checkpoints are not publicly hosted.** They were provided directly by the TrueFake
> authors, so `setup.sh` builds the environment but cannot fetch them. Place each `best.pt` in its
> baseline's directory yourself, then run the verification step below.

## Setup and running

Each baseline is set up and invoked through its own venv, following the per-directory isolation
used throughout `models/`:

```bash
cd baselines/NPR && bash setup.sh && cd ../..
baselines/NPR/venv/bin/python run.py --model tf-npr --dataset ffhq_test
```

| registry name | directory |
|---|---|
| `tf-r50e2p` | `baselines/R50_TF/` |
| `tf-r50nd` | `baselines/R50_nodown/` |
| `tf-npr` | `baselines/NPR/` |
| `tf-mislnet` | `baselines/MISLNet/` |
| `tf-clipd` | `baselines/CLIP-D/` |
| `tf-p2g` | `baselines/P2G/` |

To sweep all six against one dataset — each re-exec'd under its own venv, then cleaned up — use
`--baselines`, launched with a plain `python3` that has pandas + tqdm rather than one of the
baselines' own venvs:

```bash
python3 run.py --baselines --dataset ffhq_test --dry-run   # check the plan first
python3 run.py --baselines --dataset ffhq_test
```

`--models tf-npr,tf-clipd` does the same for a named subset.

### Verification

Each `setup.sh` runs this itself at the end, and it can be repeated by hand. Name the baseline's
**directory** as a positional argument:

```bash
baselines/NPR/venv/bin/python baselines/verify_checkpoints.py NPR
```

Verify one baseline per venv, not all six at once: the environments are deliberately split, so
P2G's venv has no `open_clip` and cannot load CLIP-D. Passing no argument checks all six and will
report spurious failures from whichever venv you happen to be in.

The check loads the checkpoint with `strict=True` and runs one forward pass, in a fresh
subprocess. This is the acceptance test for this directory and it matters more than usual here: a
wrong architecture transcription does not crash, it produces scores that look entirely reasonable
and mean nothing. Strict loading converts that into a loud failure. A baseline whose checkpoint is
absent is reported as skipped rather than failed.

## Output naming

All six share one `result_stem()` shape, `tf-<name>_GANDM`:

```
results/ffhq_test_tf-npr_GANDM_predictions.csv
```

`GANDM` rather than `GAN` or `DM` because the authors' training split spans **both** generation
families — the shared provenance described above, recorded in the filename. The schema is the
project standard; see [`results/README.md`](../results/README.md).

## Methodological notes

Three differences from the authors' own numbers, each of which affects how these columns should be
read.

**Scores are logits upstream.** The authors' `train.py` uses `BCEWithLogitsLoss` and their
`test.py` writes the raw, unbounded score. This project's contract requires a probability in
`[0, 1]`, so each `model.py` applies `sigmoid` (or `softmax(...)[1]` for MISLNet's 2-class head).
Sigmoid is monotonic, so thresholding at `0.5` here is equivalent to thresholding their raw score
at `0` — rankings, AUC and accuracy are unaffected. But the `score` column is **not numerically
identical** to the numbers in the authors' CSVs.

**Five of the six score at native resolution.** The authors' test transform is only
`ToImage → ToDtype → Normalize(ImageNet)`, with no resize and no crop. (The `CenterCrop` entries in
their pipeline are guarded by `'realFORLAB:...' in data_keys` and apply only to their FORLAB real
subset.) Two deviate, both forced by the model: **CLIP-D** resizes to 224 bicubic, center-crops,
and normalizes with CLIP's mean/std rather than ImageNet's, per its upstream config; **MISLNet**
takes 96×96 patches, discussed below. Because R50-ND and R50-E2P remove the stem stride, their
cost scales with full image area — expect them to be the slow ones on large inputs. The runner's
`--resize N` still applies on top, and every CSV records which regime produced it.

**MISLNet's architecture is reconstructed, not sourced.** Five of the six architectures come from
public code. MISLNet does not: the Stamm-lab implementation is not public, and the TrueFake
authors' released repository contains only their two ResNet-50 variants. Its architecture in
`tf_networks.py` was reverse-engineered from the checkpoint's tensor shapes.

Verified rather than assumed:

- every layer shape is forced by the checkpoint, and the module loads with `strict=True`;
- `flatten_index_permutation` is read from the checkpoint (it is `[0,1,2,3]`);
- the constrained convolution is renormalized **in `forward`**, because the stored kernels are
  demonstrably unconstrained (center taps ≈0.15, per-filter sums ≠0). Had training reprojected
  after each optimizer step, the saved weights would already satisfy the constraint.

Still inferred:

- **96×96 input**, derived from `fc_blocks.0` being `(200, 4608)` = 128×6×6 under SAME padding
  with three stride-2 pools. It is the only clean solution, and it matches the paper, which says
  MISLNet was trained with R50-E2P's augmentation — a pipeline ending in `RandomCrop(96)`.
- tanh activations and pooling placement, from the standard MISLNet topology.
- **patch aggregation.** 96×96 is far smaller than the test images, so a single crop would discard
  almost everything and make the score depend on where it landed. `model.py` tiles the image into
  non-overlapping 96×96 patches and averages the per-patch probabilities; the authors may have
  done something else. `AGGREGATE = "center"` scores one center crop instead.

**Treat `tf-mislnet` numbers with more caution than the other five.**

## Validation status

Accuracy has **not** been validated against the paper's reported figures. Table III of the TrueFake
paper gives per-generator TNR/TPR for all six on non-shared images (e.g. FFHQ TNR: NPR 1.00, CLIP-D
0.95; FLUX.1 TPR: CLIP-D 0.98, NPR 0.86). Reproducing those is the real end-to-end check on this
directory, and it requires the full TrueFake shards in `data/`.

## Implementation notes

Relevant when modifying these adapters, not when running them.

`tf_networks.py` is named that, rather than `networks.py`, because two of the vendored clones each
ship a top-level package called `networks` that a `baselines/networks.py` would shadow.

`R50_TF` and `MISLNet` vendor no upstream clone — their architectures are transcribed into
`tf_networks.py`. `run.py` records this in its `NO_CLONE_MODELS` set, so a sweep does not read a
deliberately absent clone as a broken setup.

**P2G needs an interpreter in the 3.8–3.12 range.** Its pinned `torch==2.2.2` /
`torchvision==0.17.2` / `Pillow==10.2.0` publish wheels only for those versions; past 3.12, pip
fails with `no matching distribution found for torch==2.2.2`, which reads like a bad pin rather
than too new an interpreter. These pins are a deliberate bump from upstream's `torch==2.1.0` /
`Pillow==10.0.1` (tested on 3.9), which stop at cp311 and so cannot build on the 3.12 that
miniforge currently ships; they are the earliest releases adding cp312 while still publishing
cp38. The bump was checked rather than assumed safe — the same images scored under 3.11 + torch
2.1.0 and 3.12 + torch 2.2.2 agree to ~1e-7, far below the four-decimal rounding `run.py` writes
and irrelevant at the 0.5 threshold. Its `setup.sh` selects a suitable interpreter itself, rejects
an out-of-range `PYTHON` up front, and discards a `venv/` left on the wrong interpreter by a
failed attempt. `numpy` stays pinned `<2`, as its requirements file explains.

**Never call P2G's `clip_model.encode_image()`.** Its vendored
`ResidualAttentionBlock.attention()` *stores* whatever attention mask it is handed and reads
`attn_mask=None` as "reuse the stored one" rather than "use no mask", so masks leak between calls.
Scoring takes two passes needing different mask widths — 197 tokens zero-shot, 204 with the `K=7`
learned prompts injected — so an `encode_image()` call works on the first image and fails on every
one after it with a shape mismatch. `model.py` routes the zero-shot pass through
`SliNet.extract_vector()`, which passes an explicit mask of the right width. An all-zeros mask is
numerically identical to no mask, the mask being additive, so this changes no score. Any *new*
call into the CLIP backbone hits the same trap; `encode_text()` is safe only because `load()`
calls it once, before any detection pass has run.

## Layout

```
baselines/
├── tf_networks.py           # architectures transcribed from the authors' code
├── verify_checkpoints.py    # acceptance test — every checkpoint must load strictly
└── <Detector>/
    ├── model.py             # load() / predict() — the contract run.py calls
    ├── setup.sh             # clones upstream (where needed), builds this venv
    ├── requirements.txt
    ├── venv/                # this baseline's own environment (git-ignored)
    ├── best.pt              # author-supplied checkpoint (git-ignored)
    └── <upstream clone>/    # only where the architecture is imported, not transcribed
```
