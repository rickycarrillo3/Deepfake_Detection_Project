# baselines/

The six detectors benchmarked by the TrueFake paper — Dell'Anna, Montibeller &
Boato, *"TrueFake: A Real World Case Dataset of Last Generation Fake Images
also Shared on Social Networks"*, IJCNN 2025 ([arXiv:2504.20658][paper]) —
using checkpoints supplied by the authors.

[paper]: https://arxiv.org/abs/2504.20658
[code]: https://github.com/MMLab-unitn/TrueFake-IJCNN25

## Why these live outside `models/`

Same `model.py` contract as `models/`, so the top-level `run.py` drives them
identically. They sit in their own tree because they are one coherent set
rather than six independent detectors: **all six were retrained by the TrueFake
authors on the same split** (fakes from StyleGAN2 + SDXL, reals from FFHQ +
FORLAB, 70/15/15). That shared provenance is the whole point of the set — it is
what makes their columns comparable to each other, and what makes them *not*
interchangeable with the same architectures under `models/`.

Concretely: `baselines/R50_nodown` and `models/DMimageDetection/` are the same
architecture with different weights. Mixing them up would silently compare a
TrueFake-trained detector against an original-author one. Hence the `tf-`
prefix on every registry name and every output CSV.

## The six

| registry name | dir | detector | source of architecture |
|---|---|---|---|
| `tf-r50e2p` | `R50_TF/` | **R50-E2P** — the paper's own proposed method (§IV.6): no-downsample ResNet-50 → 128-d embedding → learned-prototype head | authors' [code][code], `detector/networks.py` |
| `tf-r50nd` | `R50_nodown/` | **R50-ND** — Corvi et al., ICASSP 2023 | vendored grip-unina clone |
| `tf-npr` | `NPR/` | **NPR** — Tan et al., CVPR 2024 | vendored upstream clone |
| `tf-mislnet` | `MISLNet/` | **MISLNet** — Bayar & Stamm constrained conv, as used by Vahdati et al., CVPRW 2024 | **reconstructed from the checkpoint** — see caveat below |
| `tf-clipd` | `CLIP-D/` | **CLIP-D** — Cozzolino et al., CVPRW 2024 | vendored grip-unina clone |
| `tf-p2g` | `P2G/` | **Prompt2Guard** — Laiti et al., 2025 | vendored upstream clone |

`R50_TF` and `R50_nodown` keep the directory names the checkpoints arrived
with; R50-E2P and R50-ND are the paper's names for them.

## Layout

```
baselines/
├── tf_networks.py           # architectures transcribed from the authors' code
├── verify_checkpoints.py    # acceptance test — every checkpoint must load strictly
└── <Detector>/
    ├── model.py             # load() / predict() — the contract run.py calls
    ├── setup.sh             # clones upstream (if needed), builds this venv
    ├── requirements.txt
    ├── venv/                # this baseline's own environment (git-ignored)
    ├── best.pt              # author-supplied checkpoint (git-ignored)
    └── <upstream clone>/    # only where the architecture is imported, not transcribed
```

Every baseline owns its environment, matching the per-directory isolation used
throughout `models/`. Two directories (`R50_TF`, `MISLNet`) have no upstream
clone — their architectures are transcribed into `tf_networks.py`, since the
code they came from is either bundled in the authors' repo or not public.
`run.py` knows this via its `NO_CLONE_MODELS` set, so `--all` does not
mistake a deliberately absent clone for a broken setup.

`tf_networks.py` is named that, rather than `networks.py`, on purpose: two of
the vendored clones each ship a top-level package called `networks`, and a
`baselines/networks.py` would shadow them.

## Setup and running

Each baseline is set up and invoked through its own venv:

```bash
cd baselines/NPR && bash setup.sh
baselines/NPR/venv/bin/python run.py --model tf-npr --dataset ffhq_test
```

| baseline | setup | run |
|---|---|---|
| `tf-r50e2p` | `cd baselines/R50_TF && bash setup.sh` | `baselines/R50_TF/venv/bin/python run.py --model tf-r50e2p ...` |
| `tf-r50nd` | `cd baselines/R50_nodown && bash setup.sh` | `baselines/R50_nodown/venv/bin/python run.py --model tf-r50nd ...` |
| `tf-npr` | `cd baselines/NPR && bash setup.sh` | `baselines/NPR/venv/bin/python run.py --model tf-npr ...` |
| `tf-mislnet` | `cd baselines/MISLNet && bash setup.sh` | `baselines/MISLNet/venv/bin/python run.py --model tf-mislnet ...` |
| `tf-clipd` | `cd baselines/CLIP-D && bash setup.sh` | `baselines/CLIP-D/venv/bin/python run.py --model tf-clipd ...` |
| `tf-p2g` | `cd baselines/P2G && bash setup.sh` | `baselines/P2G/venv/bin/python run.py --model tf-p2g ...` |

`setup.sh` does **not** fetch checkpoints — they were provided directly by the
authors and are not publicly hosted. Drop each `best.pt` into its directory.

## A note on P2G's interpreter

Five baselines build against current torch. **P2G pins `torch==2.1.0` and
`Pillow==10.0.1`**, neither of which has wheels for recent Pythons — upstream
tested on 3.9. Its `setup.sh` will fail on a Python 3.12+ default. Point it at
an older interpreter:

```bash
cd baselines/P2G && PYTHON=python3.10 bash setup.sh
```

This is the clearest case for the per-directory isolation the repo already
uses: P2G genuinely cannot share an environment with the others.

## P2G: never call `clip_model.encode_image()`

P2G's vendored `ResidualAttentionBlock.attention()`
(`Prompt2Guard/src/models/clip/model.py:218`) **stores** whatever attention mask
it is handed on the module, and reads `attn_mask=None` as "reuse the stored
one" rather than "use no mask". Masks therefore leak from one call into the
next.

Scoring an image takes two passes, and they need different mask widths: the
zero-shot ImageNet pass sees `1 + 14*14 = 197` tokens, while the detection pass
injects `K=7` learned prompts for `197 + 7 = 204`. So calling `encode_image()`
(which passes `None`) for the zero-shot pass works on the *first* image and
fails on every one after it, once a detection pass has left its wider mask
behind:

```
RuntimeError: The shape of the 2D attn_mask is torch.Size([204, 204]),
but should be (197, 197).
```

The fix is to pass an explicit mask of the right width rather than `None` —
which is what upstream's oddly-named `get_none_attn_mask()` is for. `model.py`
routes the zero-shot pass through `SliNet.extract_vector()`, which does exactly
that (and applies the L2 normalization the pass needs anyway). An all-zeros
mask is numerically identical to no mask, the mask being additive, so this
changes no score.

Worth knowing when adding to this adapter: any *new* call into the CLIP
backbone has the same trap. `encode_text()` is safe only because it is called
once in `load()`, before any detection pass has run.

## Scores are logits upstream

The authors' `train.py` uses `BCEWithLogitsLoss`, and their `test.py` writes
the raw, unbounded score. This project's contract requires a probability in
`[0, 1]`, so each `model.py` applies `sigmoid` (or `softmax(...)[1]` for
MISLNet's 2-class head).

Sigmoid is monotonic, so thresholding at `0.5` here is equivalent to
thresholding their raw score at `0` — rankings, AUC and accuracy are all
unaffected. But the `score` column is **not numerically identical** to the
numbers in the authors' own CSVs. Worth knowing before comparing directly.

## Preprocessing: no resize

Five of the six score at **native resolution** — the authors' test transform
is only `ToImage → ToDtype → Normalize(ImageNet)`, with no resize and no crop.
(The `CenterCrop` entries in their pipeline are guarded by
`'realFORLAB:...' in data_keys` and apply only to their FORLAB real subset.)

Two deviate, both for reasons forced by the model:
- **CLIP-D** resizes to 224 (bicubic) + center-crops, and normalizes with
  **CLIP's** mean/std rather than ImageNet's — per its upstream config.
- **MISLNet** takes 96×96 patches; see below.

Because R50-ND and R50-E2P remove the stem stride, their cost scales with full
image area — expect them to be the slow ones on large inputs.

The runner's `--resize N` still applies on top, and every CSV records which
regime produced it.

## Caveat: MISLNet is reconstructed, not sourced

Five of the six architectures come from public code. MISLNet does not — the
Stamm-lab implementation is not public, and the TrueFake authors' released repo
contains only their two ResNet-50 variants. Its architecture in
`tf_networks.py` was reverse-engineered from the checkpoint's tensor shapes.

Verified rather than assumed:
- every layer shape is forced by the checkpoint, and the module loads with
  `strict=True`;
- `flatten_index_permutation` is read from the checkpoint (it is `[0,1,2,3]`);
- the constrained conv is renormalized **in `forward`**, because the stored
  kernels are demonstrably unconstrained (center taps ≈0.15, per-filter sums
  ≠0). Had training reprojected after each optimizer step, the saved weights
  would already satisfy the constraint.

Still inferred:
- **96×96 input**, derived from `fc_blocks.0` being `(200, 4608)` = 128×6×6
  under SAME padding with three stride-2 pools. It is the only clean solution
  and matches the paper, which says MISLNet was trained with R50-E2P's
  augmentation — a pipeline ending in `RandomCrop(96)`.
- tanh activations and pooling placement, from the standard MISLNet topology.
- **patch aggregation**: 96×96 is far smaller than the test images, so a
  single crop would discard almost everything and make the score depend on
  where it landed. `model.py` tiles the image into non-overlapping 96×96
  patches and averages the per-patch probabilities. The authors may have done
  something else. Set `AGGREGATE = "center"` to score one center crop instead.

Treat `tf-mislnet` numbers with more caution than the other five.

## Verification

```bash
baselines/venv/bin/python baselines/verify_checkpoints.py
```

Loads every checkpoint with `strict=True` and runs one forward pass. This is
the acceptance test for this directory, and it matters more than usual here: a
wrong architecture transcription does not crash, it produces scores that look
entirely reasonable and mean nothing. Strictness converts that into a loud
failure. Each baseline is checked in its own subprocess, mirroring how `run.py`
loads exactly one model per process.

**Accuracy has not yet been validated against the paper.** Table III reports
per-generator TNR/TPR for all six on non-shared images (e.g. FFHQ TNR: NPR
1.00, CLIP-D 0.95; FLUX.1 TPR: CLIP-D 0.98, NPR 0.86). Reproducing those
numbers is the real end-to-end check, and it needs real TrueFake shards in
`data/` — at time of writing it holds one placeholder image per dataset. Run it
once the shards land.
