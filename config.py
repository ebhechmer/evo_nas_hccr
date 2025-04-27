import os
import torch

# Data paths
TRAIN_DIR = os.path.join("data", "CASIA", "images", "train")
TEST_DIR  = os.path.join("data", "CASIA", "images", "test")

# Training settings
BATCH_SIZE = 64
NUM_WORKERS = 4
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
# Baseline model hyperparams
BASE_LR = 1e-3
EPOCHS = 5

PROXY_EPOCHS  = 2    # quick training per candidate
POP_SIZE       = 10  # small for initial test
NUM_GEN        = 5
MUTATION_RATE  = 0.2
CROSSOVER_RATE = 0.5
# EvoNAS proxy settings for smoke test
SMOKE_SUBSET = 500          # number of samples per individual
SMOKE_EPOCHS = 1            # epochs per individual