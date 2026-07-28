import os

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def save_checkpoint(
    model,
    optimizer,
    epoch,
    filename
):

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        filename
    )


def load_checkpoint(
    filename,
    model,
    optimizer=None
):

    checkpoint = torch.load(
        filename,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

    return checkpoint["epoch"]


def compute_metrics(
    labels,
    predictions
):

    labels = labels.cpu().numpy()

    predictions = predictions.cpu().numpy()

    predictions = predictions > 0.5

    return {

        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "precision":
            precision_score(
                labels,
                predictions
            ),

        "recall":
            recall_score(
                labels,
                predictions
            ),

        "f1":
            f1_score(
                labels,
                predictions
            )
    }


def ensure_dir(path):

    if not os.path.exists(path):
        os.makedirs(path)