"""Shared runner for every detection model in models/.

Each model exposes the contract documented in CLAUDE.md — module-level
`load()` and `predict(image_path) -> score` in its `model.py` — so one runner
drives them all. Adding a model means writing `model.py` and adding one line to
MODELS below; there is no per-model runner to copy.

Models keep their isolated venvs, so invoke this with the interpreter of the
model you are running:

    models/freqnet/venv/bin/python run.py --model freqnet --dataset ffhq_test

Writes:
    results/<model>/<dataset>.csv  (filepath, score, label)

Accuracy and inference time are printed to the console at the end of a run, but
never written to a CSV: the models time themselves differently (some not at
all), so durations are not comparable across models and should not be recorded
as if they were. The time sums the model's predict() calls only — not weight
loading, and not this runner's dataset walking or CSV writing.
"""

import argparse
import importlib.util
import os
import sys
import time
from itertools import groupby

import pandas as pd
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loaders.dataset import iter_dataset, rel_to_dataset  # noqa: E402
from metrics import print_run_summary  # noqa: E402

# Model name -> directory holding its model.py. Names match results/<model>/.
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
}

# AEROBLADE scores a whole shard per call (predict_batch) and emits extra
# per-autoencoder distance columns, so it keeps its own runner.
BESPOKE = {
    "aeroblade": "models/aeroblade/run.py",
}

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


def run(model_name, dataset_name, limit=None, force_label=None, model_kwargs=None):
    model_dir = os.path.join(REPO_ROOT, MODELS[model_name])
    module = load_model_module(model_name, model_dir)
    module.load(**(model_kwargs or {}))

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
            start = time.perf_counter()
            try:
                score = module.predict(image_path)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")
                num_skipped += 1
                continue
            # Only successful calls are added: a skipped image contributes no row, so
            # its partial work would pad the time without padding what it is over.
            elapsed += time.perf_counter() - start

            rows.append(
                {
                    "filepath": rel_to_dataset(image_path, dataset_name),
                    "score": round(float(score), 4),
                    "label": int(force_label if force_label is not None else label),
                }
            )

    if not rows:
        print(f"No images scored for dataset '{dataset_name}' — nothing to write.")
        return

    results_dir = os.path.join(REPO_ROOT, "results", model_name)
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, f"{dataset_name}.csv")
    pd.DataFrame(rows, columns=["filepath", "score", "label"]).to_csv(results_csv, index=False)

    print(f"Wrote {len(rows)} rows to {results_csv}")
    if num_skipped:
        print(f"Skipped {num_skipped} image(s)")
    print_run_summary(rows, elapsed)


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
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    args = parser.parse_args()

    if args.list:
        for name in sorted(MODELS):
            print(f"{name:16} {MODELS[name]}")
        for name, runner in sorted(BESPOKE.items()):
            print(f"{name:16} {os.path.dirname(runner)}  (bespoke runner: {runner})")
        return

    if not args.model or not args.dataset:
        parser.error("--model and --dataset are required (or use --list)")

    if args.model in BESPOKE:
        parser.error(
            f"'{args.model}' has its own runner — use: {BESPOKE[args.model]}"
        )
    if args.model not in MODELS:
        parser.error(f"unknown model '{args.model}'. Available: {', '.join(sorted(MODELS))}")

    run(
        args.model,
        args.dataset,
        limit=args.limit,
        force_label=args.force_label,
        model_kwargs=parse_model_args(args.model_arg),
    )


if __name__ == "__main__":
    main()
