"""Shared runner for every detection model in models/.

Each model exposes the contract documented in CLAUDE.md — module-level
`load()` and `predict(image_path) -> score` in its `model.py` — so one runner
drives them all. Adding a model means writing `model.py` and adding one line to
MODELS below; there is no per-model runner to copy.

Models keep their isolated venvs, so invoke this with the interpreter of the
model you are running:

    models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test

Writes:
    results/<dataset>_<model>_<family>_predictions.csv
        (filepath, score, "Predicted label", resize)

where "Predicted label" is the score thresholded to 0/1 at metrics.THRESHOLD.
The ground-truth label (0 for ffhq datasets, 1 otherwise) is not written to the
CSV; it is used only to score accuracy on the console.

where <family> is the generation family the detector was trained on (GAN/DM).
A model.py may optionally define a module-level `result_stem()` function, called
(with no arguments) after load() to override the "<model>_<family>" middle of
that filename — this is how the family is supplied, and how a model with a
--model-arg that meaningfully changes its weights (co-spy's train_dataset,
rine's ncls, aide's checkpoint) keeps those runs in separate CSVs now that all
results share one flat results/ directory. Models that don't define it fall back
to just "<model>" (no family).

Accuracy and inference time are printed to the console at the end of a run, but
never written to a CSV: the models time themselves differently (some not at
all), so durations are not comparable across models and should not be recorded
as if they were. The time sums the model's predict() calls only — not weight
loading, and not this runner's dataset walking or CSV writing.

--resize N (optional) standardizes the input across models for a controlled,
apples-to-apples comparison. When set, this runner writes a losslessly-resized
N×N PNG copy of each image to a temp dir and hands that path to the model's
predict() — so the same NxN input reaches every model *before* its own
per-model preprocessing runs, without editing any model.py. Left off (the
default), each model sees the image at its native resolution and applies only
its own built-in resizing. The resize is deliberately excluded from the timed
span (it is this runner's bookkeeping, like walking the dataset), and every
output CSV records which setting produced it in a `resize` column ("NxN" when
--resize was used this run, else "none"). Note the resampling is fixed to
bicubic here so the standardization itself is identical across models; a
model's own downstream resize (e.g. freqnet's 256) still runs on top.

--all runs every model in MODELS (and every known variant of a model, e.g.
co-spy's train_dataset=progan / sd-v1_4, or aide's three checkpoints — see
VARIANT_ARGS below) against one dataset, then deletes that model's cloned
upstream git repo AND its venv to reclaim disk space:

    python3 run.py --all --dataset ffhq_test   # see note below on which python to use

The python you launch --all with is ONLY used to run this top-level
orchestration — the loop, subprocess calls, and cleanup logic. It never
scores a single image itself. For each model, run_all() looks up THAT
model's own venv (models/<model>/venv/bin/python, or its VENV_OVERRIDE) and
re-execs `run.py --model <name> ...` as a fresh subprocess under it, so every
model's actual inference still runs fully isolated in its own venv, exactly
as if you'd invoked it directly.

Do NOT launch --all using one of the models' own venvs (e.g.
models/aide/venv/bin/python run.py --all ...) if you're deleting venvs
(default; see --keep-venv): --all will eventually try to delete that very
venv out from under the interpreter currently running it. This is detected
and skipped rather than crashing (see _is_running_interpreter_under /
_safe_rmtree below), but it means that one venv won't get cleaned up. Prefer
launching --all with a small interpreter that isn't one of the models' own —
a plain `python3` with pandas+tqdm installed is enough, since --all itself
needs nothing model-specific.

CAUTION — --all deletes cloned repos and venvs after each model finishes
(--keep-repo / --keep-venv opt out of either independently). Every model
downloads its own checkpoints fresh on each run, so that isn't a concern here.
The remaining risk is the automatic detection of each model's cloned repo
directory: it's found by locating a .git folder under the model's own
directory, deliberately NOT climbing upward, so a repo shared between two
MODELS entries (e.g. if grag2021_latent and grag2021_progan turn out to share
one clone above both of their directories) will correctly find nothing to
delete rather than guess. This search never touches this project's own
top-level .git, since that sits above every model_dir, not below it — only
nested clones strictly inside each model's own subtree are ever in scope.

Confirmed exception: grag2021_latent and grag2021_progan share one clone,
venv, and weights dir under models/DMimageDetection/ rather than each having
their own (see VENV_OVERRIDE / SHARED_CLONE_ROOTS below) — handled as a
special case, cleaned up once at the very end of the run rather than after
either model individually, since deleting it after the first would break the
second's run.

Verify with --all --dry-run first if a model's layout hasn't been confirmed.

If a model's cloned repo and/or venv is missing when --all reaches it (e.g.
right after a previous --all cleaned it up, or on a fresh checkout), its
setup.sh is run automatically to rebuild it before that model is scored,
rather than silently skipping the model. --dry-run reports this as a planned
setup step without actually running it.
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import groupby

import pandas as pd
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loaders.dataset import iter_dataset, rel_to_dataset  # noqa: E402
from metrics import THRESHOLD, print_run_summary  # noqa: E402

# Model name -> directory holding its model.py. The name also appears in each
# results CSV filename (results/<dataset>_<model>_<family>_predictions.csv).
# Listed explicitly rather than discovered: the vendored upstream clones inside
# each model directory contain their own model.py files that would be picked up
# by a filesystem walk.
MODELS = {
    "freqnet": "models/freqnet",
    "grag2021": "models/grag2021",
    "co-spy": "models/co-spy",
    "aide": "models/aide",
    "rine": "models/rine",
    "cnndetection": "models/cnndetection",
    "grag2021_latent": "models/DMimageDetection/grag2021_latent",
    "grag2021_progan": "models/DMimageDetection/grag2021_progan",
    # TrueFake baselines (baselines/, see its README). Same model.py contract
    # as models/, so the runner treats them identically; they live in their own
    # top-level directory because they are one coherent set — the six detectors
    # of Dell'Anna et al., IJCNN 2025, all retrained by those authors on the
    # same StyleGAN2+SDXL training split. The tf- prefix keeps them distinct
    # from the original-author checkpoints under models/, which share
    # architectures with some of them but not weights.
    "tf-r50nd": "baselines/R50_nodown",
    "tf-r50e2p": "baselines/R50_TF",
    "tf-npr": "baselines/NPR",
    "tf-mislnet": "baselines/MISLNet",
    "tf-clipd": "baselines/CLIP-D",
    "tf-p2g": "baselines/P2G",
}

# AEROBLADE scores a whole shard per call (predict_batch) and emits extra
# per-autoencoder distance columns, so it keeps its own runner. --all skips it
# for the same reason: its CLI contract doesn't match this runner's.
BESPOKE = {
    "aeroblade": "models/aeroblade/run.py",
}

# Extra --model-arg sets to run per model, for models with more than one
# meaningfully-different weight set. A model not listed here runs once with no
# extra args. Confirmed as of writing: co-spy (train_dataset=progan vs
# sd-v1_4) and aide (checkpoint=genimage/progan/sd14, its three shipped
# detector checkpoints) — see each model's result_stem(). Add more here as
# other multi-checkpoint models come up rather than guessing.
VARIANT_ARGS = {
    "co-spy": [
        {"train_dataset": "progan"},
        {"train_dataset": "sd-v1_4"},
    ],
    "aide": [
        {"checkpoint": "genimage"},
        {"checkpoint": "progan"},
        {"checkpoint": "sd14"},
    ],
    # Confirmed from models/rine/rine/src/utils.py: get_our_trained_model()
    # checks ncls==1/2/4 (as integers) or =="ldm", matching the four
    # checkpoints shipped in ckpt/ (model_1class_trainable.pth, etc). Values
    # here become CLI strings either way (--model-arg is always k=v text);
    # model.py's load() coerces back to int for anything other than "ldm".
    "rine": [
        {"ncls": "ldm"},
        {"ncls": "1"},
        {"ncls": "2"},
        {"ncls": "4"},
    ],
}

# Some models share one upstream clone / venv rather than each having their
# own. Confirmed: grag2021_latent and grag2021_progan both live under
# models/DMimageDetection/ alongside one shared dmimagedetection/ clone, one
# shared venv/, and one shared weights/ dir. Their own MODELS directories
# (grag2021_latent/, grag2021_progan/) contain only model.py + predict_one.py
# each — no venv or .git of their own.
VENV_OVERRIDE = {
    "grag2021_latent": "models/DMimageDetection/venv",
    "grag2021_progan": "models/DMimageDetection/venv",
}

# Models whose model.py imports no vendored upstream clone, because their
# architecture is transcribed into the repo instead (baselines/tf_networks.py).
# Without this, _repo_present() below would report "no clone found" forever and
# --all would re-run their setup.sh on every pass. Being listed here means
# "this model is not supposed to have a clone", not "its clone is missing".
NO_CLONE_MODELS = {
    "tf-r50e2p",
    "tf-mislnet",
}

# clone directory -> every model name that depends on it. Cleaned up once, at
# the very end of an --all run (after every model in MODELS has already had
# its turn) rather than after any single dependent model finishes — deleting
# it mid-run would break whichever dependent model hadn't run yet.
SHARED_CLONE_ROOTS = {
    "models/DMimageDetection/dmimagedetection": ["grag2021_latent", "grag2021_progan"],
}

# Mirrors SHARED_CLONE_ROOTS but for venvs shared across models (same
# end-of-run-only cleanup reasoning applies) — currently just the one venv
# both grag2021_latent and grag2021_progan point at via VENV_OVERRIDE.
SHARED_VENV_ROOTS = {
    "models/DMimageDetection/venv": ["grag2021_latent", "grag2021_progan"],
}

# Directory names to never descend into while looking for a model's venv
# python or its cloned repo's .git — venvs are large, irrelevant to both
# searches, and occasionally contain a stray .git from a pip vcs install that
# we do not want mistaken for the model's own clone.
_SKIP_DIRNAMES = {"venv", "site-packages", "__pycache__", "node_modules", ".git"}


def _is_venv_like(dirname):
    return dirname in _SKIP_DIRNAMES or dirname.endswith("_env")


def load_model_module(model_name, model_dir):
    """Import models/<...>/model.py as a module.

    The model dir goes on sys.path first: each model.py imports its own vendored
    upstream code, and resolves weights relative to its __file__, so it works no
    matter where this runner is invoked from.
    """
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    spec = importlib.util.spec_from_file_location(
        f"model_{model_name.replace('-', '_')}", os.path.join(model_dir, "model.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_model_args(pairs):
    """Turn --model-arg k=v pairs into kwargs forwarded to the model's load()."""
    kwargs = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--model-arg must be key=value, got: {pair!r}")
        key, value = pair.split("=", 1)
        kwargs[key] = value
    return kwargs


def _result_stem(module, model_name):
    """The '<model>_<family>' middle of this run's results filename.

    Defaults to just model_name (no family). A model.py may define a
    module-level result_stem() to override it — to append its generation family
    (GAN/DM) and/or to distinguish checkpoints of the same model (rine's ncls,
    co-spy's train_dataset, aide's checkpoint), so those runs don't overwrite
    each other now that all CSVs share one flat results/ directory."""
    override = getattr(module, "result_stem", None)
    if override is None:
        return model_name
    return override()


def _find_model_python(model_name, model_dir):
    """Every model's venv lives at <model_dir>/venv/, except models listed in
    VENV_OVERRIDE that share a venv housed elsewhere."""
    override = VENV_OVERRIDE.get(model_name)
    python_bin = (
        os.path.join(REPO_ROOT, override, "bin", "python")
        if override
        else os.path.join(model_dir, "venv", "bin", "python")
    )
    return python_bin if os.path.isfile(python_bin) else None


def _find_git_clone_roots(model_dir, max_depth=4):
    """Directories under model_dir that are themselves git clones (contain a
    .git dir), found by walking down from model_dir only — deliberately never
    climbing upward, so a repo shared above two different MODELS entries is
    left untouched (nothing found) rather than guessed at and deleted out from
    under a sibling model."""
    roots = []
    base_depth = model_dir.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(model_dir):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base_depth
        if depth >= max_depth:
            dirnames[:] = []
            continue
        if ".git" in dirnames:
            roots.append(dirpath)
            dirnames[:] = []  # don't descend into a repo we're about to delete
            continue
        dirnames[:] = [d for d in dirnames if not _is_venv_like(d)]
    return roots


def _setup_dir(model_name, model_dir):
    """Directory containing the model's setup.sh. Matches model_dir, except
    for models in VENV_OVERRIDE that share setup with a parent directory
    (e.g. grag2021_latent/progan both setup via models/DMimageDetection/)."""
    override = VENV_OVERRIDE.get(model_name)
    if override is None:
        return model_dir
    return os.path.join(REPO_ROOT, os.path.dirname(override))


def _repo_present(model_name, model_dir):
    """Whether this model's cloned upstream repo currently exists. For models
    with a shared clone (SHARED_CLONE_ROOTS), checks the shared path instead
    of model_dir, since model_dir itself never contains their .git."""
    if model_name in NO_CLONE_MODELS:
        return True
    for shared_root, dependents in SHARED_CLONE_ROOTS.items():
        if model_name in dependents:
            return os.path.isdir(os.path.join(REPO_ROOT, shared_root))
    return bool(_find_git_clone_roots(model_dir))


def _run_setup(model_name, model_dir):
    """Run this model's setup.sh to (re)build its repo clone and/or venv.
    Returns True if setup.sh was found and exited 0."""
    setup_dir = _setup_dir(model_name, model_dir)
    setup_script = os.path.join(setup_dir, "setup.sh")
    if not os.path.isfile(setup_script):
        print(f"[setup] {model_name}: no setup.sh found at {setup_script} — cannot auto-rebuild.")
        return False
    print(f"[setup] {model_name}: repo and/or venv missing — running {setup_script} ...")
    try:
        subprocess.run(["bash", setup_script], check=True, cwd=setup_dir)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[setup] {model_name}: setup.sh failed with exit {e.returncode}")
        return False


def _is_running_interpreter_under(path):
    """True if the currently running python (sys.executable) lives inside
    path — i.e. deleting path would delete the venv this very orchestrator
    process is running from. On network filesystems (NFS/Lustre, common on
    HPC clusters) this makes rmtree fail with 'Directory not empty', since the
    OS can't fully unlink a running interpreter's own open/mmapped .so files
    and silently leaves rename artifacts behind instead."""
    running = os.path.realpath(sys.executable)
    target = os.path.realpath(path)
    return running == target or running.startswith(target + os.sep)


def _safe_rmtree(path, label):
    """rmtree that skips (rather than crashes the whole --all run) when path
    houses the currently running interpreter, or when the filesystem refuses
    the delete for any other reason (e.g. NFS/Lustre rename artifacts from
    still-open files)."""
    if _is_running_interpreter_under(path):
        print(
            f"[cleanup] {label}: skipping delete of {path} — this is the venv "
            f"currently running --all itself. Launch --all with a different "
            f"interpreter (not one of the models' own venvs) to have this "
            f"cleaned up too; for now, leaving it in place."
        )
        return
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(
            f"[cleanup] {label}: failed to remove {path} ({e}) — leaving it in "
            f"place and continuing. This can happen on network filesystems "
            f"when files were recently open; a manual `rm -rf {path}` after "
            f"the run finishes should clean it up."
        )


def _resized_copy(image_path, size, dest_path):
    """Write an RGB, size×size PNG copy of image_path to dest_path, returning it.

    Used only when --resize is set, to feed every model the identical NxN input
    before its own preprocessing. PNG is lossless (no re-compression artifacts a
    frequency detector would react to) and PIL is imported lazily so runs without
    --resize keep this runner's pandas+tqdm-only footprint. Resampling is fixed to
    bicubic so the standardization is the same for all models."""
    from PIL import Image  # every model venv ships Pillow; import only when needed

    with Image.open(image_path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        img.save(dest_path, format="PNG")
    return dest_path


def run(model_name, dataset_name, limit=None, force_label=None, model_kwargs=None, resize=None):
    model_dir = os.path.join(REPO_ROOT, MODELS[model_name])
    module = load_model_module(model_name, model_dir)
    module.load(**(model_kwargs or {}))

    # When --resize is set, each image is re-encoded into this scratch dir and the
    # copy (not the original) is scored. One reused path is enough — scoring is
    # single-threaded and sequential, so each image overwrites the last. Cleaned up
    # below once scoring is done. "none" when off, "<N>x<N>" when on, recorded per row.
    resize_tag = "none" if resize is None else f"{resize}x{resize}"
    tmp_dir = tempfile.mkdtemp(prefix="dd_resize_") if resize is not None else None
    resized_path = os.path.join(tmp_dir, "resized_input.png") if tmp_dir else None

    rows = []
    num_skipped = 0

    # Summed over the predict() calls alone. Weight loading is one-off setup, and
    # walking the dataset, building rows and writing the CSV are this runner's own
    # bookkeeping — neither is model inference, so neither belongs in the total.
    elapsed = 0.0

    # One bar per shard, in order. iter_dataset() stays lazy and the dataset as a whole is
    # never materialized (at cluster scale it cannot be), but a bar needs to know how many
    # images it is counting towards before it starts, which a lazy stream cannot say. So
    # buffer a single shard at a time: that is a bounded few thousand path strings, and it
    # gives each shard's bar its total.
    for shard_dir, shard_images in groupby(
        iter_dataset(dataset_name), key=lambda pair: os.path.dirname(pair[0])
    ):
        if limit is not None and len(rows) >= limit:
            break

        shard_images = list(shard_images)
        if limit is not None:
            # Trim rather than break mid-bar, so the bar's total is what will actually be
            # scored and it still fills to 100%.
            shard_images = shard_images[: limit - len(rows)]

        for image_path, label in tqdm(
            shard_images,
            desc=f"{model_name}/{dataset_name} {os.path.basename(shard_dir)}",
            unit="img",
        ):
            try:
                # Standardize the input first when --resize is set. Done outside the
                # timed span below: it is this runner's bookkeeping, not model
                # inference, so it must not be billed to the model (same reasoning as
                # dataset walking and CSV writing).
                score_path = image_path
                if resize is not None:
                    score_path = _resized_copy(image_path, resize, resized_path)

                start = time.perf_counter()
                score = module.predict(score_path)
                # Only successful calls are added: a skipped image contributes no row,
                # so its partial work would pad the time without padding what it is over.
                elapsed += time.perf_counter() - start
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                num_skipped += 1
                continue

            # `label` is the ground truth (0 for ffhq datasets, 1 otherwise — see
            # loaders.label_for_dataset; --force-label overrides it). It is kept on
            # the row for metric computation only and is NOT written to the CSV. The
            # CSV instead carries `Predicted label`: the score thresholded to 0/1 at
            # metrics.THRESHOLD, the same cutoff print_run_summary scores accuracy
            # against, so the discrete column and the reported accuracy cannot drift.
            rows.append(
                {
                    "filepath": rel_to_dataset(image_path, dataset_name),
                    "score": round(float(score), 4),
                    "label": int(force_label if force_label is not None else label),
                    "Predicted label": int(float(score) >= THRESHOLD),
                    "resize": resize_tag,
                }
            )

    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not rows:
        print(f"No images scored for dataset '{dataset_name}' — nothing to write.")
        return

    results_dir = os.path.join(REPO_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    stem = _result_stem(module, model_name)
    results_csv = os.path.join(results_dir, f"{dataset_name}_{stem}_predictions.csv")
    pd.DataFrame(rows, columns=["filepath", "score", "Predicted label", "resize"]).to_csv(
        results_csv, index=False
    )

    print(f"Wrote {len(rows)} rows to {results_csv}")
    if num_skipped:
        print(f"Skipped {num_skipped} image(s)")
    print_run_summary(rows, elapsed)


def run_all(dataset_name, limit=None, force_label=None, keep_repo=False, keep_venv=False, dry_run=False):
    this_script = os.path.abspath(__file__)

    for model_name in sorted(MODELS):
        model_dir = os.path.join(REPO_ROOT, MODELS[model_name])
        python_bin = _find_model_python(model_name, model_dir)
        
        if not keep_repo:
            clone_dirs = _find_git_clone_roots(model_dir)
            if not clone_dirs:
                print(f"[cleanup] {model_name}: no cloned git repo found under {model_dir} — nothing to remove.")
            for clone_dir in clone_dirs:
                if dry_run:
                    print(f"[dry-run] would remove cloned repo: {clone_dir}")
                    continue
                print(
                    f"[cleanup] {model_name}: removing cloned repo {clone_dir} "
                    f"(re-run its setup.sh to restore it before using this model again)"
                )
                _safe_rmtree(clone_dir, model_name)

        if not keep_venv and model_name not in VENV_OVERRIDE:
            # VENV_OVERRIDE models' venvs are shared — handled once at the end
            # below, not per-model, for the same reason as SHARED_CLONE_ROOTS.
            venv_dir = os.path.join(model_dir, "venv")
            if os.path.isdir(venv_dir):
                if dry_run:
                    print(f"[dry-run] would remove venv: {venv_dir}")
                else:
                    print(f"[cleanup] {model_name}: removing venv {venv_dir} (re-run its setup.sh to rebuild it)")
                    _safe_rmtree(venv_dir, model_name)
                    
        needs_setup = python_bin is None or not _repo_present(model_name, model_dir)

        if needs_setup:
            if dry_run:
                print(f"[dry-run] {model_name}: repo and/or venv missing — would run its setup.sh first")
                continue
            if not _run_setup(model_name, model_dir):
                print(f"[skip] {model_name}: setup.sh missing or failed — skipping this model.")
                continue
            python_bin = _find_model_python(model_name, model_dir)
            if python_bin is None:
                print(f"[skip] {model_name}: venv still not found after setup.sh — skipping.")
                continue

        variants = VARIANT_ARGS.get(model_name, [{}])
        for variant_kwargs in variants:
            cmd = [python_bin, this_script, "--model", model_name, "--dataset", dataset_name]
            if limit is not None:
                cmd += ["--limit", str(limit)]
            if force_label is not None:
                cmd += ["--force-label", str(force_label)]
            for k, v in variant_kwargs.items():
                cmd += ["--model-arg", f"{k}={v}"]

            label = model_name if not variant_kwargs else f"{model_name} ({variant_kwargs})"
            if dry_run:
                print(f"[dry-run] would run: {' '.join(cmd)}")
                continue

            print(f"\n=== Running {label} on {dataset_name} ===")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[error] {label} exited with {e.returncode} — continuing with remaining models.")

        if not keep_repo:
            clone_dirs = _find_git_clone_roots(model_dir)
            if not clone_dirs:
                print(f"[cleanup] {model_name}: no cloned git repo found under {model_dir} — nothing to remove.")
            for clone_dir in clone_dirs:
                if dry_run:
                    print(f"[dry-run] would remove cloned repo: {clone_dir}")
                    continue
                print(
                    f"[cleanup] {model_name}: removing cloned repo {clone_dir} "
                    f"(re-run its setup.sh to restore it before using this model again)"
                )
                _safe_rmtree(clone_dir, model_name)

        if not keep_venv and model_name not in VENV_OVERRIDE:
            # VENV_OVERRIDE models' venvs are shared — handled once at the end
            # below, not per-model, for the same reason as SHARED_CLONE_ROOTS.
            venv_dir = os.path.join(model_dir, "venv")
            if os.path.isdir(venv_dir):
                if dry_run:
                    print(f"[dry-run] would remove venv: {venv_dir}")
                else:
                    print(f"[cleanup] {model_name}: removing venv {venv_dir} (re-run its setup.sh to rebuild it)")
                    _safe_rmtree(venv_dir, model_name)

    # Shared clones/venvs are handled last and only once, now that every model
    # in MODELS has already had its turn in the loop above — see
    # SHARED_CLONE_ROOTS / SHARED_VENV_ROOTS.
    for shared_root, dependent_models in SHARED_CLONE_ROOTS.items():
        if keep_repo:
            break
        shared_path = os.path.join(REPO_ROOT, shared_root)
        if not os.path.isdir(shared_path):
            print(f"[cleanup] shared root {shared_path} not present — nothing to remove.")
            continue
        if dry_run:
            print(
                f"[dry-run] would remove shared cloned repo: {shared_path} "
                f"(depended on by: {', '.join(dependent_models)})"
            )
            continue
        print(
            f"[cleanup] removing shared cloned repo {shared_path} "
            f"(depended on by: {', '.join(dependent_models)}; re-run its setup.sh "
            f"to restore it before using any of these models again)"
        )
        _safe_rmtree(shared_path, "shared")

    for shared_root, dependent_models in SHARED_VENV_ROOTS.items():
        if keep_venv:
            break
        shared_path = os.path.join(REPO_ROOT, shared_root)
        if not os.path.isdir(shared_path):
            print(f"[cleanup] shared venv {shared_path} not present — nothing to remove.")
            continue
        if dry_run:
            print(
                f"[dry-run] would remove shared venv: {shared_path} "
                f"(depended on by: {', '.join(dependent_models)})"
            )
            continue
        print(
            f"[cleanup] removing shared venv {shared_path} "
            f"(depended on by: {', '.join(dependent_models)}; re-run its setup.sh "
            f"to rebuild it before using any of these models again)"
        )
        _safe_rmtree(shared_path, "shared")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="model to run: " + ", ".join(sorted(MODELS)))
    parser.add_argument("--dataset", help="dataset_name under data/")
    parser.add_argument("--limit", type=int, help="score at most N images (quick test)")
    parser.add_argument(
        "--force-label",
        type=int,
        choices=[0, 1],
        help="override the ground-truth label (e.g. 1 for a fake-only dataset)",
    )
    parser.add_argument(
        "--model-arg",
        action="append",
        metavar="KEY=VALUE",
        help="passed through to the model's load(), e.g. --model-arg train_dataset=sd-v1_4",
    )
    parser.add_argument(
        "--resize",
        type=int,
        metavar="N",
        help="standardize input: feed every model an NxN (bicubic) copy of each image "
        "before its own preprocessing, for a controlled comparison. Recorded in the "
        "output CSV's 'resize' column. Off by default (native resolution).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run every model (and known variant) against --dataset, then delete each "
        "model's cloned upstream repo to reclaim disk space; see module docstring",
    )
    parser.add_argument(
        "--keep-repo",
        action="store_true",
        help="with --all, skip deleting cloned repos after each model finishes",
    )
    parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="with --all, skip deleting venvs after each model finishes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="with --all, print what would run and what would be deleted, without doing either",
    )
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(MODELS):
            print(f"{name:16} {MODELS[name]}")
        for name, runner in sorted(BESPOKE.items()):
            print(f"{name:16} {os.path.dirname(runner)}  (bespoke runner: {runner})")
        return

    if args.all:
        if args.model:
            parser.error("--all runs every model; don't pass --model with it")
        if not args.dataset:
            parser.error("--dataset is required with --all")
        run_all(
            args.dataset,
            limit=args.limit,
            force_label=args.force_label,
            keep_repo=args.keep_repo,
            keep_venv=args.keep_venv,
            dry_run=args.dry_run,
        )
        return

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (or use --all, or --list)")

    if args.model in BESPOKE:
        parser.error(
            f"'{args.model}' has its own runner — use: {BESPOKE[args.model]}"
        )
    if args.model not in MODELS:
        parser.error(f"unknown model '{args.model}'. Available: {', '.join(sorted(MODELS))}")

    if args.resize is not None and args.resize <= 0:
        parser.error(f"--resize must be a positive integer, got {args.resize}")

    run(
        args.model,
        args.dataset,
        limit=args.limit,
        force_label=args.force_label,
        model_kwargs=parse_model_args(args.model_arg),
        resize=args.resize,
    )


if __name__ == "__main__":
    main()