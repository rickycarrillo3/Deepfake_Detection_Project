# baselines/

Detectors from the **TrueFake** paper ([Dell'Anna, Montibeller & Boato, IJCNN
2025](https://arxiv.org/abs/2504.20658)), re-run on our dataset shards to reproduce the paper's
numbers. Kept separate from `models/` — which holds the detectors *we* are benchmarking for the
correlation/ensembling analysis — so a reproduction run is never mistaken for one of ours.

## Relationship to `models/`

Everything about the contract is identical. Same layout, same `model.py` interface, same top-level
`run.py`. Read [`models/README.md`](../models/README.md) for the full contract; it applies here
unchanged.

The **only** difference is the results filename:

```
results/baseline_<dataset_name>_<model_name>_<family>_predictions.csv
```

`run.py` adds the `baseline_` prefix automatically for any model whose registry entry points into
`baselines/`, so it cannot drift from where the code actually lives. Column schema is unchanged
(`filepath`, `score`, `Predicted label`, `resize`).

## Layout

```
baselines/<model_name>/
├── model.py              # load() / predict() — the same contract run.py calls
├── setup.sh              # clones upstream, builds the venv, fetches weights
├── requirements.txt      # this baseline's own dependencies
├── venv/                 # isolated environment, git-ignored
├── weights/              # checkpoint(s), git-ignored
└── ...                   # vendored clone of the upstream repo, git-ignored
```

## Running

Same as any model — invoke `run.py` with that baseline's own venv:

```bash
baselines/r50_e2p/venv/bin/python run.py --model r50_e2p --dataset ffhq_test
```

Baselines are included in `run.py --list` and in `--all` sweeps alongside the `models/` entries.

## What's here

| Baseline | Paper name | Status |
| --- | --- | --- |
| `r50_e2p` | R50-E2P | implemented — pretrained weights published upstream |

### The paper's other detectors

The paper's Table III/IV compare six detectors: **R50-E2P**, **R50-ND**, **MISLNet**, **NPR**,
**CLIP-D** and **P2G**. Only R50-E2P lives in the TrueFake repo — it is the authors' own proposed
method, and the only one whose weights they publish. The other five were trained "using the code
and pipeline provided in their respective papers", so reproducing them means pulling five separate
upstream repos, and the authors do not distribute their TrueFake-trained checkpoints.

Note also that **R50-ND is Gragnaniello et al. 2021** — already implemented in this repo as
`models/grag2021` (and `models/DMimageDetection/grag2021_*`). Its numbers here would come from the
original authors' checkpoints, not from TrueFake-trained ones, so it is *not* a like-for-like
reproduction of the paper's R50-ND column.
