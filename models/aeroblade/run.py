"""
Runner for the AEROBLADE model.

Usage:
    PYTHONPATH=<repo_root> aeroblade_env/bin/python run.py <dataset_name>

AEROBLADE batches most efficiently when given a whole directory per call
(see model.py's module docstring — each directory passed to
compute_distances() gets one batched reconstruction pass per AE; each
individual file gets its own). So this groups iter_dataset()'s
(image_path, label) stream by shard directory and calls
model.predict_batch(shard_dir) once per shard.

Writes:
    results/aeroblade/<dataset_name>.csv           (filepath, score, label,
                                                      + one raw-distance
                                                      column per AE)
    results/aeroblade/<dataset_name>_summary.csv    (avg per-image time)
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loaders.dataset import iter_dataset, rel_to_dataset  # noqa: E402
from metrics import print_run_summary  # noqa: E402

import model as aeroblade_model  # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "results", "aeroblade")


def run(dataset_name: str):
    aeroblade_model.load()

    # Group ground-truth labels by shard directory. We still go through
    # iter_dataset() for the label rule + partial-shard tolerance, but
    # score each shard directory in a single predict_batch() call.
    shards = defaultdict(dict)  # shard_dir -> {image_path: label}
    for image_path, label in iter_dataset(dataset_name):
        shards[os.path.dirname(image_path)][image_path] = label

    results = []
    total_time = 0.0
    total_images = 0

    for shard_dir, label_map in shards.items():
        shard_name = os.path.basename(shard_dir)

        start = time.time()
        try:
            scored = aeroblade_model.predict_batch(shard_dir)
            #print("=== SCORED KEYS ===")
            #for k in scored.keys():
            #    print(repr(k))
        except Exception as e:
            print(f"Skipping shard {shard_name}: {e}")
            continue
        elapsed = time.time() - start

        for image_path, label in label_map.items():
            key = f"{Path(image_path).parent.name}/{Path(image_path).name}"
            #print("LOOKING FOR:", repr(key)) 
            info = scored.get(key)
            if info is None:
                print(f"No score returned for: {image_path}")
                continue
            rel_path = rel_to_dataset(image_path, dataset_name)
            row = {"filepath": rel_path, "score": round(info["score"], 4), "label": label}
            row.update({k: v for k, v in info.items() if k != "score"})
            results.append(row)

        total_time += elapsed
        total_images += len(label_map)
        print(f"Finished folder: {shard_name}")

    if total_images == 0:
        print(f"No images found for dataset '{dataset_name}' — nothing to write.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.DataFrame(results)
    results_csv = os.path.join(RESULTS_DIR, f"{dataset_name}.csv")
    df.to_csv(results_csv, index=False)

    avg_time = total_time / total_images
    summary_csv = os.path.join(RESULTS_DIR, f"{dataset_name}_summary.csv")
    pd.DataFrame([{
        "dataset": dataset_name,
        "total_images": total_images,
        "avg_seconds_per_image": avg_time,
    }]).to_csv(summary_csv, index=False)

    print(
        f"\nFinished\n\n"
        f"Total images: {total_images}\n"
        f"Average processing time per image: {avg_time:.4f} seconds\n\n"
        f"CSV saved as {results_csv}\n"
        f"Summary saved as {summary_csv}"
    )
    # total_time sums the predict_batch calls only, so it excludes model loading
    # and the shard grouping — the same span run.py times for every other model.
    print_run_summary(results, total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AEROBLADE over a dataset of shards.")
    parser.add_argument("dataset_name", help="Name of the dataset under data/, e.g. sd3_faces_shards")
    args = parser.parse_args()
    run(args.dataset_name)
