import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

from model import Grag2021Model

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, PROJECT_ROOT)
from loaders.dataset import iter_dataset  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "grag2021")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset_name under data/")
    args = parser.parse_args()

    pairs = list(iter_dataset(args.dataset))
    print(f"Found {len(pairs)} images in data/{args.dataset}")

    model = Grag2021Model()
    model.load()

    rows = []
    for image_path, label in tqdm(pairs, desc=args.dataset):
        score = model.predict(image_path)
        rows.append({"filepath": image_path, "score": round(score, 4), "label": label})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"{args.dataset}.csv"), index=False)
    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
