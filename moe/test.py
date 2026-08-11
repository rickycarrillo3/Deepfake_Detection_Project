import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config

from datasets import ExpertDataset

from models import (
    ImageMoE,
    GatingNetwork,
    CNNEncoder,
)

from utils import (
    compute_metrics,
    load_checkpoint
)


# --------------------------------------------------
# Build the exact same model as train.py
# --------------------------------------------------

def create_model():

    encoder = CNNEncoder(
        feature_dim=config.CNN_FEATURE_DIM
    )

    gating = GatingNetwork(
        image_encoder=encoder,
        feature_dim=config.CNN_FEATURE_DIM,
        num_experts=config.NUM_EXPERTS
    )

    model = ImageMoE(
        gating_network=gating,
        num_experts=config.NUM_EXPERTS,
        hidden_dims=config.HIDDEN_DIMS
    )

    return model




# --------------------------------------------------
# Test
# --------------------------------------------------

@torch.no_grad()
def test(
    model,
    loader
):

    model.eval()

    all_labels = []
    all_probs = []

    total_loss = 0.0

    criterion = torch.nn.BCEWithLogitsLoss()


    for images, scores, labels in loader:

        images = images.to(
            config.DEVICE
        )

        scores = scores.to(
            config.DEVICE
        )

        labels = labels.to(
            config.DEVICE
        ).unsqueeze(1)


        # Forward pass only
        logits = model(
            images,
            scores
        )


        loss = criterion(
            logits,
            labels
        )


        probs = torch.sigmoid(
            logits
        )


        total_loss += loss.item()


        all_labels.append(
            labels
        )

        all_probs.append(
            probs
        )


    labels = torch.cat(
        all_labels
    )

    probs = torch.cat(
        all_probs
    )


    # Calculate metrics
    metrics = compute_metrics(
        labels,
        probs
    )


    metrics["loss"] = (
        total_loss /
        len(loader)
    )


    return metrics


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print(
        "Testing saved model..."
    )


    # --------------------------------------------------
    # Transform
    # --------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(
            (
                config.IMAGE_SIZE,
                config.IMAGE_SIZE
            )
        ),

        transforms.ToTensor()

    ])


    # --------------------------------------------------
    # Test dataset
    # --------------------------------------------------

    test_dataset = ExpertDataset(
        config.TEST_CSV_PATH,
        config.IMAGE_ROOT,
        transform
    )


    print(
        "Test samples:",
        len(test_dataset)
    )


    # --------------------------------------------------
    # Test loader
    # --------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )


    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = create_model()

    model.to(
        config.DEVICE
    )


    # --------------------------------------------------
    # Load saved checkpoint
    # --------------------------------------------------

    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR,
        config.SAVE_NAME
    )


    print(
        "Loading checkpoint:",
        checkpoint_path
    )


    load_checkpoint(
        checkpoint_path,
        model,
    )


    # --------------------------------------------------
    # Test model
    # --------------------------------------------------

    metrics = test(
        model,
        test_loader
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("================================")
    print("TEST RESULTS")
    print("================================")

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )


    print("================================")


if __name__ == "__main__":

    main()