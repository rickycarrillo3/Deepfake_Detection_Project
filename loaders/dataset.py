import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Image file extensions a shard may contain, matched case-insensitively. JPEG is
# included because TrueFake's social-network-shared subsets are JPEG, not PNG —
# a PNG-only filter would walk those datasets and yield nothing at all.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def label_for_dataset(dataset_name):
    return 0 if "ffhq" in dataset_name.lower() else 1


def rel_to_dataset(image_path, dataset_name, data_dir=DATA_DIR):
    """Return an image path relative to its dataset dir, e.g. 'shard_01/00042.png'.

    iter_dataset yields absolute paths (needed to open the file); use this to derive the
    compact, portable form stored in results CSVs.
    """
    dataset_dir = os.path.join(data_dir, dataset_name)
    return os.path.relpath(image_path, dataset_dir)


def iter_dataset(dataset_name, data_dir=DATA_DIR):
    """Yield (image_path, label) pairs for data/<dataset_name>/shard_XX/ images.

    Files are included by extension (IMAGE_EXTENSIONS). Labels are derived from dataset_name (see
    data/README.md). Tolerates partial shard uploads — only shards currently present on disk are
    iterated. Paths are absolute, lazily yielded.
    """
    label = label_for_dataset(dataset_name)
    dataset_dir = os.path.join(data_dir, dataset_name)

    for shard in sorted(os.listdir(dataset_dir)):
        shard_path = os.path.join(dataset_dir, shard)
        if not shard.startswith("shard_") or not os.path.isdir(shard_path):
            continue
        for fname in sorted(os.listdir(shard_path)):
            if fname.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(shard_path, fname), label
