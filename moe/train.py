import os

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    random_split
)

from torchvision import transforms

from tqdm import tqdm


from datasets import ExpertDataset

from models import (
    ImageMoE,
    GatingNetwork
)

from utils import (
    save_checkpoint,
    compute_metrics,
    ensure_dir,
    load_checkpoint
)


import config


# -----------------------------
# Build model
# -----------------------------

from models import CNNEncoder


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
    
# -----------------------------
# Training epoch
# -----------------------------


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()


    total_loss = 0


    loop = tqdm(loader)


    for images, scores, labels in loop:


        images = images.to(
            config.DEVICE
        )

        scores = scores.to(
            config.DEVICE
        )

        labels = labels.to(
            config.DEVICE
        ).unsqueeze(1)



        optimizer.zero_grad()



        with torch.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            logits = model(
                images,
                scores
            )


            loss = criterion(
                logits,
                labels
            )



        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )

        scaler.update()



        total_loss += loss.item()


        loop.set_description(
            f"loss {loss.item():.4f}"
        )


    return (
        total_loss /
        len(loader)
    )



# -----------------------------
# Validation
# -----------------------------


@torch.no_grad()

def validate(
    model,
    loader,
    criterion
):

    model.eval()


    losses = []

    all_labels = []

    all_probs = []



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



        losses.append(
            loss.item()
        )


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


    metrics = compute_metrics(
        labels,
        probs
    )


    metrics["loss"] = (
        sum(losses) /
        len(losses)
    )


    return metrics



# -----------------------------
# Main
# -----------------------------


def main():

    ensure_dir(
        config.CHECKPOINT_DIR
    )


    transform = transforms.Compose([

        transforms.Resize(
            (
                config.IMAGE_SIZE,
                config.IMAGE_SIZE
            )
        ),

        transforms.ToTensor()

    ])



    dataset = ExpertDataset(
        config.CSV_PATH,
        config.IMAGE_ROOT,
        transform
    )



    train_size = int(
        len(dataset) *
        (1-config.VAL_SPLIT)
    )


    val_size = (
        len(dataset)
        -
        train_size
    )


    train_set, val_set = random_split(
        dataset,
        [
            train_size,
            val_size
        ]
    )



    train_loader = DataLoader(
        train_set,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )


    val_loader = DataLoader(
        val_set,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )



    model = create_model()

    model.to(
        config.DEVICE
    )



    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )


    criterion = nn.BCEWithLogitsLoss()



    scaler = torch.cuda.amp.GradScaler()



    checkpoint_path = os.path.join(
    config.CHECKPOINT_DIR,
    config.SAVE_NAME)

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        saved_epoch = load_checkpoint(
            checkpoint_path,
            model,
            optimizer)["epoch"]
        start_epoch = saved_epoch + 1
        print(
            f"Resuming from epoch {start_epoch + 1}"
        )

    else:
        print(
            "No checkpoint found. Starting from scratch."
        )
        start_epoch = 0
        
    best_loss = float("inf")
    patience_counter = 0
    



    for epoch in range(
        start_epoch,
        config.EPOCHS
    ):


        print(
            f"\nEpoch {epoch+1}/{config.EPOCHS}"
        )


        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )


        metrics = validate(
            model,
            val_loader,
            criterion
        )


        print(
            "Train loss:",
            train_loss
        )

        print(
            metrics
        )



        if metrics["loss"] < best_loss:


            best_loss = metrics["loss"]

            patience_counter = 0


            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(
                    config.CHECKPOINT_DIR,
                    config.SAVE_NAME
                )
            )


            print(
                "Saved best model"
            )


        else:

            patience_counter += 1


        if patience_counter >= config.PATIENCE:

            print(
                "Early stopping"
            )

            break




if __name__ == "__main__":

    main()