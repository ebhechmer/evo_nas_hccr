import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import TRAIN_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR

def get_hwdb_loaders():
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    test_ds  = datasets.ImageFolder(TEST_DIR,  transform=transform)

    train_ld = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR
    )
    test_ld = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR
    )
    return train_ld, test_ld