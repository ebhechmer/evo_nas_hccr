import json, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Subset, DataLoader
import argparse

from config import DEVICE, BASE_LR, EPOCHS, BATCH_SIZE, NUM_WORKERS
from src.data.loader import get_hwdb_loaders
from src.models.baseline import BaselineCNN

cudnn.benchmark = True

def train_one_epoch(model, loader, optimizer, scaler, epoch, log_interval=50):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader, 1):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            preds = model(imgs)
            loss  = F.cross_entropy(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * imgs.size(0)
        total_correct += (preds.argmax(1) == labels).sum().item()
        total         += imgs.size(0)

        if batch_idx % log_interval == 0:
            batch_acc = total_correct/total
            print(f"[Epoch {epoch}] Batch {batch_idx}/{len(loader)} "
                  f"loss={loss.item():.4f} acc={batch_acc:.4f}")

    avg_loss = total_loss / total
    avg_acc  = total_correct / total
    print(f"[Epoch {epoch}] TRAIN → avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}")
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
    print(f"             VAL   → avg_acc={acc:.4f}")
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only first N samples")
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    print("==> Loading data…")
    train_ld, test_ld = get_hwdb_loaders()
    print(f"    Full sizes → Train: {len(train_ld.dataset)}, Test: {len(test_ld.dataset)}")

    if args.subset:
        print(f"==> Subsetting to first {args.subset} samples")
        train_ds = Subset(train_ld.dataset, range(min(args.subset, len(train_ld.dataset))))
        test_ds  = Subset(test_ld.dataset,  range(min(args.subset, len(test_ld.dataset))))
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
        test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
        print(f"    Subsets → Train: {len(train_ld.dataset)}, Test: {len(test_ld.dataset)}")

    num_classes = len(train_ld.dataset.classes)
    model       = BaselineCNN(num_classes).to(DEVICE)
    optimizer   = optim.Adam(model.parameters(), lr=BASE_LR)
    scaler      = GradScaler()
    print("Baseline Params:", sum(p.numel() for p in model.parameters()))
    train_losses, train_accs, val_accs = [], [], []
    for epoch in range(1, EPOCHS+1):
        print(f"\n### Epoch {epoch}/{EPOCHS}")
        tl, ta = train_one_epoch(model, train_ld, optimizer, scaler, epoch, args.log_interval)
        va     = evaluate(model, test_ld)
        train_losses.append(tl); train_accs.append(ta); val_accs.append(va)

    # 1) save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baseline_cnn.pth")
    print("→ Saved BaselineCNN to checkpoints/baseline_cnn.pth")

    # 2) save history
    with open("checkpoints/baseline_history.json", "w") as f:
        json.dump({
            "train_loss": train_losses,
            "train_acc":  train_accs,
            "val_acc":    val_accs
        }, f)
    print("→ Saved history to checkpoints/baseline_history.json")
    
if __name__ == "__main__":
    main()