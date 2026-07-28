import torch

#########################################
# DATA
#########################################

CSV_PATH = "moe_data/train.csv"

IMAGE_ROOT = "moe_data/images"

#########################################
# MODEL
#########################################

NUM_EXPERTS = 3

CNN_FEATURE_DIM = 512

HIDDEN_DIMS = [32, 16]

#########################################
# TRAINING
#########################################

BATCH_SIZE = 32

EPOCHS = 25

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 4

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#########################################
# CHECKPOINTS
#########################################

CHECKPOINT_DIR = "checkpoints"

SAVE_NAME = "best_model.pt"

# Validation split
VAL_SPLIT = 0.2

# Training
PATIENCE = 5

# Image preprocessing
IMAGE_SIZE = 224

# Logging
PRINT_EVERY = 20