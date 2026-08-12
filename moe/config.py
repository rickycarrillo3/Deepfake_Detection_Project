import torch

#########################################
# DATA
#########################################

CSV_PATH = "moe_data/data.csv"
TRAIN_CSV_PATH = "moe_data/train.csv"
TEST_CSV_PATH = "moe_data/test.csv"

IMAGE_ROOT = "moe_data/images"

#########################################
# MODEL
#########################################

NUM_EXPERTS = 3

EXPERTS = [
    "rine_DM",
    "dmimagedetection_GAN",
    "co-spy_DM"
]

## CNN_FEATURE_DIM = 512
CNN_FEATURE_DIM = 256

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

PATIENCE = 5


#########################################
# CHECKPOINTS
#########################################

CHECKPOINT_DIR = "checkpoints"

SAVE_NAME = "best_model.pt"

#########################################
# SPLITS
#########################################
VAL_SPLIT = 0.2

TEST_SPLIT = 0.3

TTS_SEED = None

# Image preprocessing
## IMAGE_SIZE = 224
IMAGE_SIZE = 1024


# Logging
PRINT_EVERY = 20