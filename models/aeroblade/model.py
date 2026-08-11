"""
AEROBLADE model adapter — implements the models/<model_name>/ contract:
    load()                    -> prepare for inference
    predict(image_path)       -> float, the raw reconstruction distance

Wraps jonasricker/aeroblade (vendored in ./aeroblade/), specifically
aeroblade.high_level_funcs.compute_distances().

What this model outputs, and why its CSV differs
------------------------------------------------
AEROBLADE is training-free: it reconstructs each image with several
pretrained latent-diffusion autoencoders (AEs) and measures the LPIPS
reconstruction error between original and reconstruction. Images that
are reconstructed more accurately are more likely to be outputs of an
LDM sharing that AE.

The raw distances are *negative* LPIPS errors, so larger (less negative)
means an easier reconstruction, i.e. more likely diffusion-generated.
compute_distances(..., compute_max=True) — the default — already adds a
`repo_id="max"` row per file holding the best-AE distance; predict()
returns that value unchanged.

That value is a reconstruction error, not a probability: the paper uses
these distances only for ranking (AP/ROC), never against a fixed
real/fake cutoff. So this model.py declares, for the shared run.py:
    SCORE_COLUMN     = "reconstruction score"  (not "score")
    PREDICTED_LABEL  = False                   (no thresholded 0/1 column)
    SUPPORTS_RESIZE  = False                   (see the caching note below)
plus extra_columns(), which adds one raw-distance column per AE — kept
for the correlation analysis, since disagreement between AEs may itself
be informative.

Batching: why predict() caches per directory
------------------------------------------------
compute_distances()'s loop is `for dir in dirs: ds = ImageFolder(dir);
for repo_id in repo_ids: compute_reconstructions(ds, repo_id, ...)`.
Each *directory* passed gets one batched reconstruction pass per AE;
each individual *file* passed gets its own pass. Scoring image-by-image
would therefore reload every AE pipeline per image.

So predict() keeps run.py's one-image-at-a-time contract but memoizes by
parent directory: the first image of a shard triggers a single
compute_distances() call over that whole shard, and every other image in
it is a cache hit. Only the current directory's results are held, so
memory stays bounded across a dataset. AEROBLADE requires all images in
one directory to share dimensions — shards already satisfy this.

This caching is also why SUPPORTS_RESIZE is False: run.py's --resize
path rewrites every image to one reused temp filename, so a
directory-keyed cache would hand the first image's scores to the entire
run. run.py rejects --resize for this model rather than compute
something silently wrong.

Disk
------------------------------------------------
This is by far the most disk-hungry model here, and left at upstream's defaults
it will exhaust a cluster quota part-way through a dataset. Two separate caches
grow per image, per autoencoder (measured at 1024x1024):

    reconstruction PNG          1.5 MB
    joblib spatial-LPIPS entry  4.6 MB

With the two autoencoders configured below that is ~12 MB per image: ~12 GB per
1000-image shard, ~240 GB for a 20-shard dataset. Neither is needed once the
score reaches the results CSV, so both are OFF by default here — reconstructions
are written under a per-process directory and deleted after each shard, and the
distance memoization is unwrapped entirely. Re-enable either with
--model-arg keep_reconstructions=true / cache_distances=true, which buys faster
re-runs over the same images at that price in disk.

Separately, the weights themselves are ~10 GB in the HuggingFace cache, because
AutoPipelineForImage2Image pulls each *whole* pipeline (Kandinsky's 4.6 GB unet,
SD's 1.6 GB unet, both text encoders) when AEROBLADE only ever runs the
autoencoders (160 MB + 259 MB). That is upstream's loading path, not something
this adapter chooses; point HF_HOME at scratch space rather than a quota'd home
directory.

Devices
------------------------------------------------
Upstream resolves the device itself via aeroblade.misc.device() ("cuda"
if torch.cuda.is_available(), else "cpu"), so the CUDA path here is
upstream's own: fp16 pipelines, enable_model_cpu_offload(), and
torch.compile. On a CPU-only machine those three break outright, so
load() installs a compatibility shim — see _patch_for_cpu() — that is
inert whenever CUDA is present.
"""

import atexit
import contextlib
import os
import shutil
from pathlib import Path

import torch

_MODEL_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_ROOT = _MODEL_DIR / "aeroblade_output"


@contextlib.contextmanager
def _cwd(path):
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


# aeroblade.distances and aeroblade.complexities both build a
# joblib.Memory(location="cache") at *import* time, which joblib creates
# immediately, relative to the CWD — the repo root, since run.py imports this
# from there. Import them from inside aeroblade_output/ so that directory lands
# under this model instead of next to run.py. load() then repoints where entries
# are actually written too (_patch_cache_location), which is what matters if a
# future upstream resolves the location lazily rather than eagerly.
_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
with _cwd(_OUTPUT_ROOT):
    from aeroblade.high_level_funcs import compute_distances

# --- Contract flags read by the shared run.py (see module docstring) -------

SCORE_COLUMN = "reconstruction score"
PREDICTED_LABEL = False
SUPPORTS_RESIZE = False
SCORE_PRECISION = 6

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

# compute_distances() writes a reconstructed copy of every image here, one per
# autoencoder — see the disk note in the module docstring.
_RECON_ROOT = _OUTPUT_ROOT / "reconstructions"

# Upstream's joblib cache of computed distances — see _patch_distance_cache().
_CACHE_ROOT = _OUTPUT_ROOT / "cache"

_batch_size = BATCH_SIZE
_num_workers = NUM_WORKERS

# Disk policy, both off by default — see the module docstring. Overridable with
# --model-arg keep_reconstructions=true / cache_distances=true.
_keep_reconstructions = False
_cache_distances = False

# Where this process writes reconstructions when it is going to delete them
# again: its own subdirectory, so cleaning it up cannot disturb another
# aeroblade process sharing the same node. Set in load().
_run_recon_root = None

# Directory-keyed score cache, holding one directory at a time (see docstring).
_cached_dir = None
_cached_scores = {}


def _memoized_funcs(module):
    """The @mem.cache-decorated functions in an upstream module."""
    import joblib

    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, joblib.memory.MemorizedFunc):
            yield name, obj


def _patch_distance_cache(enabled):
    """Decide where — or whether — upstream memoizes computed distances.

    aeroblade.distances and aeroblade.complexities each declare
    Memory(location="cache"): relative to the *current working directory*, which
    is the repo root since run.py imports this model.py from there. Every
    model.py resolves paths from __file__ instead (see models/README.md).

    Disabled (the default) it is not a matter of *where*: that cache is by far
    the largest thing this model writes — a spatial LPIPS map per image per
    autoencoder, ~4.6 MB each at 1024x1024, ~9 GB per 1000-image shard — and it
    only saves recomputing a number that already went to the results CSV. Unwrap
    the decorated functions so calls go straight through and nothing is stored.
    """
    import joblib

    import aeroblade.complexities as ae_complexities
    import aeroblade.distances as ae_distances

    modules = (ae_distances, ae_complexities)

    if not enabled:
        for module in modules:
            for name, func in _memoized_funcs(module):
                setattr(module, name, func.func)  # the undecorated original
        # joblib created the (empty) tree eagerly at import; take it back out.
        shutil.rmtree(_CACHE_ROOT, ignore_errors=True)
        return

    memory = joblib.Memory(location=str(_CACHE_ROOT), compress=("lz4", 9), verbose=0)
    for module in modules:
        module.mem = memory
        for _, func in _memoized_funcs(module):
            func.store_backend = memory.store_backend


def _patch_noninteractive():
    """Stop upstream from blocking on stdin mid-run.

    compute_reconstructions() calls aeroblade.misc.safe_mkdir(), which prompts
    "Directory ... exists, continue? (y/n)" and exit()s on anything else. It is
    only reached when the cached reconstruction directory is *incomplete* (the
    complete case short-circuits earlier), i.e. left behind by an interrupted
    run, so recomputing into it is the right answer — but a batch job on a
    cluster has no stdin to answer with and would die there. Answer it "yes"
    unconditionally.
    """
    import aeroblade.image as ae_image

    ae_image.safe_mkdir = lambda directory: directory.mkdir(parents=True, exist_ok=True)


def _patch_for_cpu():
    """Make upstream's CUDA-only assumptions survive a CPU-only machine.

    Called only when no CUDA device is present, so a GPU run is byte-for-byte
    upstream behaviour. compute_reconstructions() in aeroblade/src/aeroblade/
    image.py hardcodes three things that fail without a GPU:

      1. torch_dtype=torch.float16 — CPU convolutions have no Half kernel.
      2. pipe.enable_model_cpu_offload() — diffusers raises without an
         accelerator to offload from.
      3. torch.compile(ae) — unnecessary here, and a common source of
         backend failures on macOS.
    """
    import aeroblade.image as ae_image

    original_from_pretrained = ae_image.AutoPipelineForImage2Image.from_pretrained

    def from_pretrained_fp32(repo_id, *args, **kwargs):
        # Only the dtype changes. upstream's variant="fp16" stays: on the Hub
        # these repos ship safetensors *only* under the fp16 variant (the plain
        # revision is .bin), and use_safetensors=True would fail without it.
        # diffusers casts those fp16 files up to float32 on load.
        kwargs["torch_dtype"] = torch.float32
        pipe = original_from_pretrained(repo_id, *args, **kwargs)
        pipe.enable_model_cpu_offload = lambda *a, **kw: None
        return pipe

    ae_image.AutoPipelineForImage2Image.from_pretrained = staticmethod(from_pretrained_fp32)
    ae_image.torch.compile = lambda module, *a, **kw: module


def _as_bool(value):
    """--model-arg values arrive as strings, so accept the usual spellings."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _hf_cache_dir():
    """Where the ~10 GB of autoencoder weights will land, for the load banner."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return HF_HUB_CACHE
    except Exception:  # older/newer hub layouts — the banner is not worth failing over
        return os.environ.get("HF_HOME", "~/.cache/huggingface")


def load(
    device=None,
    batch_size=None,
    num_workers=None,
    keep_reconstructions=None,
    cache_distances=None,
):
    """Prepare for inference. There is no persistent model object —
    compute_distances() loads the AE pipelines itself, per call.

    device: "cuda"/"cpu" to force one (via CUDA_VISIBLE_DEVICES for "cpu");
    default follows torch.cuda.is_available(), as upstream does.
    batch_size / num_workers: forwarded to compute_distances(), for tuning
    against a particular GPU's memory.
    keep_reconstructions / cache_distances: keep the two on-disk caches instead
    of discarding them. Both default to off — see the disk note in the module
    docstring; together they write ~12 MB per 1024x1024 image, which is what
    exhausts a cluster quota part-way through a dataset.
    """
    global _batch_size, _num_workers, _keep_reconstructions, _cache_distances
    global _run_recon_root

    if device is not None and str(device).lower().startswith("cpu"):
        # aeroblade.misc.device() only consults torch.cuda.is_available(), so
        # hiding the GPUs is the way to force CPU through the whole stack.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if batch_size is not None:
        _batch_size = int(batch_size)
    if num_workers is not None:
        _num_workers = int(num_workers)
    if keep_reconstructions is not None:
        _keep_reconstructions = _as_bool(keep_reconstructions)
    if cache_distances is not None:
        _cache_distances = _as_bool(cache_distances)

    _patch_distance_cache(_cache_distances)
    _patch_noninteractive()

    resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved == "cpu":
        _patch_for_cpu()

    if _keep_reconstructions:
        # Stable path, so a later run reuses these instead of recomputing them.
        _run_recon_root = _RECON_ROOT
    else:
        # Private to this process, so deleting it after each shard cannot pull
        # the ground out from under another aeroblade job on the same node.
        _run_recon_root = _RECON_ROOT / f"run_{os.getpid()}"
        atexit.register(shutil.rmtree, _run_recon_root, True)
    _run_recon_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[aeroblade] device={resolved} batch_size={_batch_size} "
        f"num_workers={_num_workers} autoencoders={len(AUTOENCODERS)}"
    )
    kept = [
        name
        for name, on in (("reconstructions", _keep_reconstructions), ("distances", _cache_distances))
        if on
    ]
    growth = (
        "discarding both caches, ~0 GB growth"
        if not kept
        else f"KEEPING {' + '.join(kept)} — grows ~{6.1 * len(kept):.1f} MB per 1024px image per AE"
    )
    print(f"[aeroblade] disk: {growth}")
    print(f"[aeroblade] weights cache: {_hf_cache_dir()} (~10 GB; set HF_HOME to move it)")


def result_stem():
    """No generation-family suffix: AEROBLADE is a reconstruction-distance
    method, not a classifier trained on one generator family."""
    return "aeroblade"


def _short_ae_name(repo_id: str) -> str:
    return repo_id.split("/")[-1]


def _extract_scores(distances):
    """Pivot compute_distances()'s long-format DataFrame (columns: dir,
    image_size, repo_id, transform, distance_metric, file, distance) into
    {filename: {"score": best_distance, <ae_short_name>: raw_distance, ...}},
    keyed by bare filename (the caller already knows the directory)."""
    results = {}
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

        results[Path(file_path).name] = {"score": best_distance, **per_ae}
    return results


def predict_batch(shard_dir):
    """Score every image in one directory in a single compute_distances()
    call — one batched reconstruction pass per AE (see module docstring).

    Returns: {filename: {"score": raw_max_distance, <ae_short_name>: raw_distance, ...}}
    """
    if _run_recon_root is None:
        raise RuntimeError("Call load() before scoring.")

    distances = compute_distances(
        dirs=[Path(shard_dir)],
        transforms=["clean"],
        repo_ids=AUTOENCODERS,
        distance_metrics=[DISTANCE_METRIC],
        amount=None,
        reconstruction_root=_run_recon_root,
        seed=SEED,
        batch_size=_batch_size,
        num_workers=_num_workers,
        compute_max=True,
    )
    results = _extract_scores(distances)

    if not _keep_reconstructions:
        # The distances are extracted by now, so the reconstructed PNGs (one per
        # image per autoencoder, ~1.5 MB each at 1024x1024) have served their
        # purpose. Dropping them per shard is what keeps a dataset-sized run
        # flat on disk instead of growing by ~3 GB per 1000-image shard. The
        # cost is that re-running recomputes them on the GPU; pass
        # --model-arg keep_reconstructions=true when that trade is wrong.
        shutil.rmtree(_run_recon_root, ignore_errors=True)
        _run_recon_root.mkdir(parents=True, exist_ok=True)

    return results


def _info_for(image_path):
    """Cached per-image record, scoring the whole parent directory on first
    use. Holds one directory at a time: moving to the next shard drops the
    previous one rather than accumulating a dataset's worth of rows."""
    global _cached_dir, _cached_scores

    image_path = Path(image_path)
    shard_dir = str(image_path.parent)

    if shard_dir != _cached_dir:
        _cached_scores = predict_batch(shard_dir)
        _cached_dir = shard_dir

    info = _cached_scores.get(image_path.name)
    if info is None:
        raise KeyError(f"AEROBLADE returned no distance for {image_path}")
    return info


def predict(image_path) -> float:
    """Raw best-AE reconstruction distance (negative LPIPS). Not a
    probability and not in [0, 1] — see the module docstring."""
    return _info_for(image_path)["score"]


def extra_columns(image_path):
    """Per-autoencoder raw distances, written as extra CSV columns by run.py."""
    return {k: v for k, v in _info_for(image_path).items() if k != "score"}
