"""
Runner for the Grag2021_progan model.

Usage:
    PYTHONPATH=<repo_root> ../dmimagedetection_env/bin/python run.py <dataset_name>

Writes:
    results/grag2021_progan/<dataset_name>.csv
    results/grag2021_progan/<dataset_name>_summary.csv
"""

import argparse
import os
import sys
import time

import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from loaders.dataset import iter_dataset  # noqa: E402

import model as grag_model  # noqa: E402

# Named after this directory (models/DMimageDetection/grag2021_progan/)
# so the sibling grag2021_latent/run.py is identical apart from its own
# folder name.
MODEL_NAME = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_REPO_ROOT, "results", MODEL_NAME)


def run(dataset_name: str):
    grag_model.load()

    results = []
    total_time = 0.0
    total_images = 0
    last_shard = None

    for image_path, label in iter_dataset(dataset_name):
        shard = os.path.basename(os.path.dirname(image_path))
        if shard != last_shard:
            if last_shard is not None:
                print(f"Finished folder: {last_shard}")
            last_shard = shard

        start = time.time()
        try:
            score = grag_model.predict(image_path)
        except Exception as e:
            print(f"Skipping corrupted image: {image_path}\n{e}")
            continue
        elapsed = time.time() - start

        total_time += elapsed
        total_images += 1
        results.append({"filepath": f"{shard}/{os.path.basename(image_path)}", "score": score, "label": label})

    if last_shard is not None:
        print(f"Finished folder: {last_shard}")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Run {MODEL_NAME} over a dataset of shards.")
    parser.add_argument("dataset_name", help="Name of the dataset under data/, e.g. sd3_faces_shards")
    args = parser.parse_args()
    run(args.dataset_name)
