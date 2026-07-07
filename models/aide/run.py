import argparse
import os
import sys
import time

import pandas as pd
from tqdm import tqdm

from model import AIDEModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from loaders.dataset import iter_dataset  # noqa: E402

MODEL_NAME = "aide"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", MODEL_NAME)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="dataset_name under data/")
    parser.add_argument(
        "--force-label",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional override. Use --force-label 1 for fake-only datasets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional quick-test limit before running a full dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional path to AIDE .pth checkpoint. Otherwise uses AIDE_WEIGHTS_PATH "
            "or the default GenImage_train.pth path."
        ),
    )
    args = parser.parse_args()

    pairs = list(iter_dataset(args.dataset))
    if args.limit is not None:
        pairs = pairs[: args.limit]

    print(f"Found {len(pairs)} images in data/{args.dataset}")

    model = AIDEModel(checkpoint_path=args.checkpoint)
    model.load()

    rows = []
    start = time.time()

    for image_path, label in tqdm(pairs, desc=args.dataset):
        score = model.predict(image_path)
        final_label = args.force_label if args.force_label is not None else label
        rows.append(
            {
                "filepath": image_path,
                "score": round(float(score), 6),
                "label": int(final_label),
            }
        )

    elapsed = time.time() - start
    num_images = len(rows)
    avg_seconds_per_image = elapsed / num_images if num_images else 0.0

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f"{args.dataset}.csv")
    summary_csv = os.path.join(RESULTS_DIR, f"{args.dataset}_summary.csv")

    # Main result CSV: keep exactly the requested schema/order.
    result_df = pd.DataFrame(rows, columns=["filepath", "score", "label"])
    result_df.to_csv(out_csv, index=False)

    # Companion timing CSV for the results/ convention.
    summary_df = pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "dataset": args.dataset,
                "num_images": num_images,
                "total_seconds": round(elapsed, 6),
                "avg_seconds_per_image": round(avg_seconds_per_image, 6),
                "results_csv": out_csv,
            }
        ]
    )
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote {num_images} rows to {out_csv}")
    print(f"Wrote timing summary to {summary_csv}")
    print(f"Average processing time: {avg_seconds_per_image:.4f} sec/image")


if __name__ == "__main__":
    main()
