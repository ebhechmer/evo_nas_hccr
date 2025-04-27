import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from config import BASE_LR, EPOCHS, DEVICE, BATCH_SIZE, NUM_WORKERS
from src.data.loader import get_hwdb_loaders
from src.models.baseline import BaselineCNN

def train_one_epoch(model, loader, optimizer, epoch, log_interval=50):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss   += loss.item() * imgs.size(0)
        total_correct+= (preds.argmax(1) == labels).sum().item()
        total        += imgs.size(0)

        if batch_idx % log_interval == 0:
            print(f"  [Epoch {epoch}] Batch {batch_idx}/{len(loader)} — "
                  f"Batch loss {loss.item():.4f}, "
                  f"Batch acc {preds.argmax(1).eq(labels).sum().item()/imgs.size(0):.4f}")

    avg_loss = total_loss / total
    avg_acc  = total_correct / total
    print(f"[Epoch {epoch}] TRAIN → Avg loss {avg_loss:.4f}, Avg acc {avg_acc:.4f}")
    return avg_loss, avg_acc

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    acc = correct / total
    print(f"               VAL   → Avg acc {acc:.4f}")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Baseline HCCR Training")
    parser.add_argument("--subset", type=int, default=None,
                        help="If set, only use the first N samples for train and test")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Batches between progress logs")
    args = parser.parse_args()

    print("==> Loading data loaders…")
    train_ld, test_ld = get_hwdb_loaders(BATCH_SIZE, NUM_WORKERS)
    print(f"    Full dataset sizes → Train: {len(train_ld.dataset)}, Test: {len(test_ld.dataset)}")

    if args.subset:
        print(f"==> Subsetting to first {args.subset} samples")
        train_ds = Subset(train_ld.dataset, list(range(min(args.subset, len(train_ld.dataset)))))
        test_ds  = Subset(test_ld.dataset,  list(range(min(args.subset, len(test_ld.dataset)))))
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
        test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        print(f"    Subset sizes → Train: {len(train_ld.dataset)}, Test: {len(test_ld.dataset)}")

    num_classes = len(train_ld.dataset.dataset.classes) if isinstance(train_ld.dataset, Subset) else len(train_ld.dataset.classes)
    print(f"==> Building model (device={DEVICE}) with {num_classes} classes…")
    model = BaselineCNN(num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n### Starting Epoch {epoch}/{EPOCHS}")
        train_one_epoch(model, train_ld, optimizer, epoch, log_interval=args.log_interval)
        evaluate(model, test_ld)

if __name__ == "__main__":
    main()