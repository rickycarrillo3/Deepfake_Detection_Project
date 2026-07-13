"""
AEROBLADE model adapter — implements the models/<model_name>/ contract:
    load()                    -> load weights, prepare model for inference
    predict(image_path)       -> float in [0, 1], predicted fake-probability

Wraps jonasricker/aeroblade (vendored in ./aeroblade/), specifically
aeroblade.high_level_funcs.compute_distances().

IMPORTANT — read before running at scale
------------------------------------------
AEROBLADE is training-free: it reconstructs each image with several
pretrained latent-diffusion autoencoders (AEs) and measures the LPIPS
reconstruction error between original and reconstruction. Images that
are reconstructed more accurately (smaller error) are more likely to be
outputs of an LDM sharing that AE.

compute_distances()'s loop structure is `for dir in dirs: ds =
ImageFolder(dir); for repo_id in repo_ids: compute_reconstructions(ds,
repo_id, ...)`. That means each *directory* you pass gets batched
through a single reconstruction pass per AE (via batch_size/num_workers);
each individual *file* you pass gets its own pass. So: predict_batch()
takes a whole shard directory (matching data/<dataset>/shard_XX/) and
scores every image in it in one call — that's the efficient path, and
what run.py uses. predict(image_path) passes a single file and is kept
only for contract-compatibility; it reloads all AE pipelines on every
call, so avoid it in a loop.

score semantics
------------------------------------------
AEROBLADE's raw distances are *negative* LPIPS distances (less negative,
i.e. larger, for the best-matching AE => easier reconstruction => more
likely fake). compute_distances(..., compute_max=True) — the default —
already adds a `repo_id="max"` row per file with the max (best-AE)
distance; we take that value and pass it through a sigmoid purely to
land in [0, 1] for the results/ schema. This is a monotonic transform,
so rank order and correlation with other models' scores are unaffected
— the absolute value is not a calibrated "probability of fake" (the
paper only uses these distances for ranking / AP, not a fixed threshold).
"""

import math
import os
from pathlib import Path

from aeroblade.high_level_funcs import compute_distances

# --- Config (mirrors scripts/run_aeroblade.py's defaults) ------------------

AUTOENCODERS = [
    "CompVis/stable-diffusion-v1-1",        # SD1
#    "stabilityai/stable-diffusion-2-base",  # SD2
    "kandinsky-community/kandinsky-2-1",    # KD2.1
]
DISTANCE_METRIC = "lpips_vgg_2"  # best-performing layer per the paper
BATCH_SIZE = 4
NUM_WORKERS = 2
SEED = 1

# compute_distances() caches reconstructions to disk itself and skips
# recomputation on repeat runs against the same images/AEs.
_RECON_ROOT = (
    Path(os.path.dirname(os.path.abspath(__file__))) / "aeroblade_output" / "reconstructions"
)


def load():
    """No persistent model object to hold onto — compute_distances() loads
    the AE pipelines itself, per call. Kept for contract compatibility."""
    _RECON_ROOT.mkdir(parents=True, exist_ok=True)


def _short_ae_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def _reduce_to_score(distance: float) -> float:
    """Squash a raw (negative) LPIPS distance into [0, 1]. Monotonic only —
    not a calibrated fake-probability, see module docstring."""
    return 1.0 / (1.0 + math.exp(-distance * 10))


def _extract_scores(distances):
    """Pivot compute_distances()'s long-format DataFrame (columns: dir,
    image_size, repo_id, transform, distance_metric, file, distance) into
    {image_path: {"score": ..., <ae_short_name>: raw_distance, ...}}."""
    results = {}
    #for file_path in distances["file"].unique():
    #    print(repr(file_path))
    for file_path, group in distances.groupby("file"):
        per_ae = {
            _short_ae_name(row.repo_id): float(row.distance)
            for row in group.itertuples()
            if row.repo_id != "max"
        }
        max_row = group[group["repo_id"] == "max"]
        if not max_row.empty:
            best_distance = float(max_row["distance"].iloc[0])
        else:
            best_distance = max(per_ae.values(), default=float("-inf"))

        key = f"{Path(file_path).parent.name}/{Path(file_path).name}"

        results[key] = {
            "score": _reduce_to_score(best_distance),
            **per_ae,
        }
    return results


def predict_batch(shard_dir):
    """Score every image in a single shard directory in one
    compute_distances() call — the efficient path; use this from run.py
    instead of looping predict(). All images in a directory must share
    the same dimensions (an AEROBLADE requirement; shards already
    satisfy this).

    Returns: {image_path: {"score": float, <ae_short_name>: raw_distance, ...}}
    """
    distances = compute_distances(
        dirs=[Path(shard_dir)],
        transforms=["clean"],
        repo_ids=AUTOENCODERS,
        distance_metrics=[DISTANCE_METRIC],
        amount=None,
        reconstruction_root=_RECON_ROOT,
        seed=SEED,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        compute_max=True,
    )
    results = _extract_scores(distances)

    shard = Path(shard_dir).name

    results = {
            f"{shard}/{filename.lstrip('/')}": info
            for filename, info in results.items()
    }

    return results
    
    #return _extract_scores(distances)


def predict(image_path: str) -> float:
    """Single-image convenience wrapper for contract compatibility.
    Passes a single file as its own 'dir', which reloads all AE
    pipelines on every call — see module docstring. Prefer
    predict_batch(shard_dir) for anything beyond a handful of images."""
    distances = compute_distances(
        dirs=[Path(image_path)],
        transforms=["clean"],
        repo_ids=AUTOENCODERS,
        distance_metrics=[DISTANCE_METRIC],
        amount=None,
        reconstruction_root=_RECON_ROOT,
        seed=SEED,
        batch_size=1,
        num_workers=0,
        compute_max=True,
    )
    return _extract_scores(distances)[str(image_path)]["score"]
