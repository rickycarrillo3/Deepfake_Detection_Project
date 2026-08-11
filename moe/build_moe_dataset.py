import argparse
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from tqdm import tqdm

import config


# --------------------------------------------------
# Dataset prefixes considered real
# --------------------------------------------------

REAL_DATASET_PREFIXES = {
    "ffhq",
}


# --------------------------------------------------
# Dataset label
# --------------------------------------------------

def get_dataset_label(dataset):
    """
    Returns:

        0 = real
        1 = fake
    """

    dataset_lower = dataset.lower()

    for prefix in REAL_DATASET_PREFIXES:

        if dataset_lower.startswith(prefix):

            return 0

    return 1


# --------------------------------------------------
# Parse result filename
# --------------------------------------------------

def parse_result_filename(
    filename,
    known_datasets
):
    """
    Example:

        ffhq_faces_aide_DM_predictions.csv

    Returns:

        dataset:
            ffhq_faces

        model:
            aide_DM
    """

    stem = filename.replace(
        "_predictions.csv",
        ""
    )

    matches = [
        dataset
        for dataset in known_datasets
        if stem.startswith(
            dataset + "_"
        )
    ]

    if not matches:

        raise ValueError(
            f"Could not identify dataset from {filename}"
        )

    dataset = max(
        matches,
        key=len
    )

    model = stem[
        len(dataset) + 1:
    ]

    return dataset, model


# --------------------------------------------------
# Find image
# --------------------------------------------------

def find_image(
    data_path,
    dataset,
    filepath
):
    """
    Resolves:

        data_path/
            dataset/
                filepath
    """

    image_path = (
        Path(data_path)
        / dataset
        / filepath
    )

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    return image_path


# --------------------------------------------------
# Build dataset
# --------------------------------------------------

def build_dataset(
    results_path,
    data_path,
    output_path,
    selected_datasets=None,
    selected_models=None
):

    results_path = Path(
        results_path
    )

    data_path = Path(
        data_path
    )

    output_path = Path(
        output_path
    )


    #
    # Find available datasets
    #

    known_datasets = [
        x.name
        for x in data_path.iterdir()
        if x.is_dir()
    ]


    print("\nAvailable datasets:")

    for dataset in sorted(
        known_datasets
    ):

        print(
            f"  {dataset}"
        )


    #
    # Find prediction CSVs
    #

    result_files = list(
        results_path.glob(
            "*_predictions.csv"
        )
    )


    if not result_files:

        raise RuntimeError(
            "No *_predictions.csv files found"
        )


    datasets = {}


    #
    # Group CSVs by dataset
    #

    for csv_file in result_files:

        dataset, model = (
            parse_result_filename(
                csv_file.name,
                known_datasets
            )
        )


        if selected_datasets:

            if dataset not in selected_datasets:

                continue


        if selected_models:

            if model not in selected_models:

                continue


        datasets.setdefault(
            dataset,
            []
        ).append(
            (
                model,
                csv_file
            )
        )


    if not datasets:

        raise RuntimeError(
            "No datasets/models matched your filters"
        )


    #
    # Process datasets
    #

    for dataset, experts in datasets.items():

        print(
            "\n" + "=" * 60
        )

        print(
            f"Building dataset: {dataset}"
        )


        dataset_output = (
            output_path /
            dataset
        )


        image_output = (
            dataset_output /
            "images"
        )


        image_output.mkdir(
            parents=True,
            exist_ok=True
        )


        merged = None


        #
        # Load expert scores
        #

        for model_name, csv_file in experts:

            print(
                f"Loading expert: {model_name}"
            )


            df = pd.read_csv(
                csv_file
            )


            required = {
                "filepath",
                "score"
            }


            missing = (
                required -
                set(df.columns)
            )


            if missing:

                raise ValueError(
                    f"{csv_file} missing columns {missing}"
                )


            df = df[
                [
                    "filepath",
                    "score"
                ]
            ]


            df = df.rename(
                columns={
                    "score": model_name
                }
            )


            if merged is None:

                merged = df

            else:

                merged = merged.merge(
                    df,
                    on="filepath",
                    how="inner"
                )


        if (
            merged is None
            or merged.empty
        ):

            print(
                "No overlapping images between experts"
            )

            continue


        #
        # Link images
        #

        output_names = []

        valid_rows = []


        for idx, row in tqdm(
            merged.iterrows(),
            total=len(merged),
            desc=f"Linking {dataset}"
        ):

            filepath = row[
                "filepath"
            ]


            try:

                source = find_image(
                    data_path,
                    dataset,
                    filepath
                )


            except FileNotFoundError:

                continue


            safe_name = (
                filepath
                .replace("/", "_")
                .replace("\\", "_")
            )


            new_name = (
                f"{dataset}_{safe_name}"
            )


            destination = (
                image_output /
                new_name
            )


            if destination.exists():

                destination.unlink()


            #
            # Create hard link instead
            # of copying the image
            #

            destination.hardlink_to(
                source
            )


            output_names.append(
                new_name
            )


            valid_rows.append(
                idx
            )


        #
        # Remove missing images
        #

        merged = (
            merged
            .loc[valid_rows]
            .reset_index(drop=True)
        )


        #
        # Add image column
        #

        merged.insert(
            0,
            "image",
            output_names
        )


        #
        # Remove original filepath
        #

        merged = merged.drop(
            columns=[
                "filepath"
            ]
        )


        #
        # Add label
        #

        label = get_dataset_label(
            dataset
        )


        merged.insert(
            1,
            "label",
            label
        )


        #
        # Save per-dataset CSV
        #
        # This is an intermediate CSV.
        # The final train/test split happens
        # after all datasets are merged.
        #

        output_csv = (
            dataset_output /
            "train.csv"
        )


        merged.to_csv(
            output_csv,
            index=False
        )


        print(
            f"Saved dataset CSV: {output_csv}"
        )


        print(
            f"Images linked: {len(output_names)}"
        )


        print(
            "Experts used:"
        )


        for expert in merged.columns[2:]:

            print(
                f"  {expert}"
            )


# --------------------------------------------------
# Merge all datasets and split train/test
# --------------------------------------------------

def merge_all_datasets(
    output_path
):

    output_path = Path(
        output_path
    )


    #
    # Final shared image directory
    #

    combined_image_dir = (
        output_path /
        "images"
    )


    combined_image_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    #
    # Find per-dataset CSVs
    #

    all_csvs = list(
        output_path.glob(
            "*/train.csv"
        )
    )


    if not all_csvs:

        raise RuntimeError(
            "No dataset CSVs found"
        )


    dfs = []


    #
    # Merge datasets
    #

    for csv_file in all_csvs:

        dataset_dir = (
            csv_file.parent
        )


        image_dir = (
            dataset_dir /
            "images"
        )


        df = pd.read_csv(
            csv_file
        )


        new_images = []


        for img in df["image"]:

            source = (
                image_dir /
                img
            )


            destination = (
                combined_image_dir /
                img
            )


            #
            # Create symbolic link to
            # the dataset-specific image
            #

            if not destination.exists():

                destination.symlink_to(
                    source.resolve()
                )


            new_images.append(
                img
            )


        df["image"] = new_images


        dfs.append(
            df
        )


    #
    # Combine all datasets
    #

    merged = pd.concat(
        dfs,
        ignore_index=True
    )


    print()
    print(
        "=" * 60
    )

    print(
        "Combined dataset size:",
        len(merged)
    )


    #
    # Train/test split
    #
    # config.TEST_SPLIT controls the
    # percentage placed into the test set.
    #
    # Example:
    #
    # TEST_SPLIT = 0.20
    #
    # gives:
    #
    # 80% train
    # 20% test
    #

    try:

        train_df, test_df = (
            train_test_split(
                merged,
                test_size=config.TEST_SPLIT,
                random_state= config.TTS_SEED,
                shuffle=True,
                stratify=merged["label"]
            )
        )

    except ValueError:

        #
        # Fallback if stratification is not
        # possible, e.g. only one class exists.
        #

        print(
            "Warning: Could not stratify split."
        )

        train_df, test_df = (
            train_test_split(
                merged,
                test_size=config.TEST_SPLIT,
                random_state=42,
                shuffle=True
            )
        )


    #
    # Create train/test directories
    #

    # train_dir = (
    #    output_path /
    #    "train"
    # )


    #test_dir = (
    #    output_path /
    #    "test"
    #)


    #train_dir.mkdir(
    #    parents=True,
    #    exist_ok=True
    #)


    #test_dir.mkdir(
    #    parents=True,
    #    exist_ok=True
    #)


    #
    # Save train CSV
    #

    # train_csv = (
    #    output_path /
    #    "train.csv"
    #)


    merged.to_csv(
        config.CSV_PATH,
        index=False
    )

    train_df.to_csv(
        config.TRAIN_CSV_PATH,
        index=False
    )


    #
    # Save test CSV
    #

    #test_csv = (
    #    output_path /
    #    "test.csv"
    #)


    test_df.to_csv(
        config.TEST_CSV_PATH,
        index=False
    )


    #
    # Print results
    #

    print()
    print(
        "Train/test split complete."
    )

    print()

    print(
        f"TEST_SPLIT: {config.TEST_SPLIT}"
    )

    print(
        f"Train samples: {len(train_df)}"
    )

    print(
        f"Test samples: {len(test_df)}"
    )

    print()

    print(
        f"Train percentage: "
        f"{len(train_df) / len(merged):.2%}"
    )

    print(
        f"Test percentage: "
        f"{len(test_df) / len(merged):.2%}"
    )

    print()
    
    print(
        f"Full CSV: {config.CSV_PATH}"
    )

    print(
        f"Train CSV: {config.TRAIN_CSV_PATH}"
    )

    print(
        f"Test CSV: {config.TEST_CSV_PATH}"
    )
    
    
    print(
        f"Images: {combined_image_dir}"
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build MoE dataset from detector "
            "prediction CSVs"
        )
    )


    parser.add_argument(
        "--results_path",
        required=True,
        help=(
            "Directory containing "
            "*_predictions.csv files"
        )
    )


    parser.add_argument(
        "--data_path",
        required=True,
        help="Root dataset directory"
    )


    parser.add_argument(
        "--output_path",
        default="moe_data",
        help="Output directory"
    )


    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Datasets to include. "
            "Default: all datasets"
        )
    )


    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Experts/models to include. "
            "Default: all models"
        )
    )


    args = parser.parse_args()


    #
    # Build individual datasets
    #

    build_dataset(
        results_path=args.results_path,
        data_path=args.data_path,
        output_path=args.output_path,
        selected_datasets=args.datasets,
        selected_models=args.models
    )


    #
    # Merge everything and create
    # final train/test split
    #

    merge_all_datasets(
        args.output_path
    )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()