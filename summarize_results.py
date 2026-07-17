from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")

TRAINING_TYPE = {
    "aide": "GAN",
    "cnndetection": "GAN",
    "co-spy_progan": "GAN",
    "co-spy_sd-v1_4": "DM",
    "freqnet": "GAN",
    "grag2021": "GAN",
    "grag2021_latent": "DM",
    "grag2021_progan": "GAN",
    "rine": "DM",
}

summary = []

for csv_file in RESULTS_DIR.rglob("*.csv"):
    folder = csv_file.parent.name

    # Ignore folders not in the mapping
    if folder not in TRAINING_TYPE:
        continue

    # Rename models in the output
    if folder in {"grag2021_progan", "grag2021_latent"}:
        model = "DMimageDetection"
    elif folder in {"co-spy_progan", "co-spy_sd-v1_4"}:
        model = "co-spy"
    else:
        model = folder

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()

    # Skip files with unexpected columns
    required = {"score", "label"}
    if not required.issubset(df.columns):
        print(f"Skipping {csv_file}")
        print("Columns:", df.columns.tolist())
        continue

    # Convert scores to binary predictions
    pred = (df["score"] >= 0.5).astype(int)

    # Metrics
    accuracy = (pred == df["label"]).mean()
    avg_error = (df["score"] - df["label"]).abs().mean()

    # Dataset name (remove trailing "_faces" if present)
    dataset = csv_file.stem
    if dataset.endswith("_faces"):
        dataset = dataset[:-6]

    training = TRAINING_TYPE[folder]

    standardized_name = (
        f"{dataset}_{model}_{training}_predictions.csv"
    )

    summary.append({
        "model_name": standardized_name,
        "dataset": dataset,
        "model": model,
        "training_data_type": training,
        "accuracy": accuracy,
        "average_error": avg_error,
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(
    ["dataset", "model", "training_data_type"]
)

summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)

print(summary_df)
print(f"\nProcessed {len(summary_df)} CSV files.")
print(f"Saved results to {RESULTS_DIR / 'summary.csv'}")