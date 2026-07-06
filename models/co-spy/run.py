import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from model import CoSpyModel

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_ROOT)
from loaders.dataset import iter_dataset  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "co-spy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset_name under data/")
    parser.add_argument("--train-dataset", default="sd-v1_4", help="CO-SPY checkpoint set to load")
    args = parser.parse_args()

    pairs = list(iter_dataset(args.dataset))
    print(f"Found {len(pairs)} images in data/{args.dataset}")

    model = CoSpyModel(train_dataset=args.train_dataset)
    model.load()

    rows = []
    total_time = 0.0
    for image_path, label in tqdm(pairs, desc=args.dataset):
        start = time.time()
        score = model.predict(image_path)
        total_time += time.time() - start
        rows.append({"filepath": image_path, "score": round(score, 4), "label": label})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"{args.dataset}.csv"), index=False)

    avg_time = total_time / len(rows) if rows else 0.0
    pd.DataFrame([{"num_images": len(rows), "avg_seconds_per_image": round(avg_time, 4)}]).to_csv(
        os.path.join(RESULTS_DIR, f"{args.dataset}_summary.csv"), index=False
    )
    print(f"Wrote {len(rows)} rows, avg {avg_time:.4f}s/image")


if __name__ == "__main__":
    main()
