import os
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data paths
TRAIN_DIR = os.path.join("data", "CASIA", "images", "train")
TEST_DIR  = os.path.join("data", "CASIA", "images", "test")

# DataLoader settings
BATCH_SIZE         = 512         # as large as fits in GPU memory
NUM_WORKERS        = 8
PIN_MEMORY         = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR    = 2

# Baseline training
BASE_LR = 1e-3
EPOCHS  = 5

# EvoNAS proxy settings
SMOKE_SUBSET   = 5000   # use 5k samples per candidate
SMOKE_EPOCHS   = 2      # train each proxy 2 epochs
POP_SIZE       = 20
NUM_GEN        = 10
MUTATION_RATE  = 0.2
CROSSOVER_RATE = 0.5