import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Dataset prefixes considered real
REAL_DATASET_PREFIXES = {
    "ffhq",
}


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


def parse_result_filename(filename, known_datasets):
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
        if stem.startswith(dataset + "_")
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


def find_image(data_path, dataset, filepath):
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


def build_dataset(
    results_path,
    data_path,
    output_path,
    selected_datasets=None,
    selected_models=None
):

    results_path = Path(results_path)
    data_path = Path(data_path)
    output_path = Path(output_path)


    #
    # Find available datasets
    #

    known_datasets = [
        x.name
        for x in data_path.iterdir()
        if x.is_dir()
    ]


    print("\nAvailable datasets:")

    for dataset in sorted(known_datasets):
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

        dataset, model = parse_result_filename(
            csv_file.name,
            known_datasets
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

        print("\n" + "=" * 60)
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


        if merged is None or merged.empty:

            print(
                "No overlapping images between experts"
            )

            continue


        #
        # Link images and skip missing files
        #

        output_names = []
        valid_rows = []


        for idx, row in tqdm(
            merged.iterrows(),
            total=len(merged),
            desc=f"Linking {dataset}"
        ):
            filepath = row["filepath"]


            try:

                source = find_image(
                    data_path,
                    dataset,
                    filepath
                )


            except FileNotFoundError:

               # print(
               #     f"Warning: missing image, skipping: "
              #      f"{dataset}/{filepath}"
              #  )

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


            # Create hard link instead of copying
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
        # Remove rows where images were missing
        #

        merged = (
            merged
            .loc[valid_rows]
            .reset_index(drop=True)
        )


        #
        # Create training CSV
        #

        merged.insert(
            0,
            "image",
            output_names
        )


        merged = merged.drop(
            columns=[
                "filepath"
            ]
        )


        label = get_dataset_label(
            dataset
        )


        merged.insert(
            1,
            "label",
            label
        )


        output_csv = (
            dataset_output /
            "train.csv"
        )


        merged.to_csv(
            output_csv,
            index=False
        )


        print(
            f"Saved CSV: {output_csv}"
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
            

def merge_all_datasets(output_path):

    output_path = Path(output_path)

    combined_image_dir = (
        output_path /
        "images"
    )

    combined_image_dir.mkdir(
        exist_ok=True
    )


    all_csvs = list(
        output_path.glob(
            "*/train.csv"
        )
    )


    dfs = []


    for csv_file in all_csvs:

        dataset_dir = csv_file.parent

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


            if not destination.exists():

                destination.symlink_to(
                    source.resolve()
                )


            new_images.append(
                img
            )


        df["image"] = new_images

        dfs.append(df)



    merged = pd.concat(
        dfs,
        ignore_index=True
    )


    merged.to_csv(
        output_path / "train.csv",
        index=False
    )


    print(
        f"Final dataset size: {len(merged)}"
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build MoE dataset from detector prediction CSVs"
        )
    )


    parser.add_argument(
        "--results_path",
        required=True,
        help="Directory containing *_predictions.csv files"
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


    build_dataset(
        results_path=args.results_path,
        data_path=args.data_path,
        output_path=args.output_path,
        selected_datasets=args.datasets,
        selected_models=args.models
    )
    
    
    merge_all_datasets(
        args.output_path
    )



if __name__ == "__main__":

    main()