#!/usr/bin/env python3
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from config import DEVICE, BATCH_SIZE, NUM_WORKERS, BASE_LR, EPOCHS
from data.loader import get_hwdb_loaders
from evo.evo_nas import EvoCNN

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for imgs, labels in loader:
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

    return total_loss/total, total_correct/total

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return correct/total

def main():
    cudnn.benchmark = True
    # 1. Load full data
    train_ld, test_ld = get_hwdb_loaders()
    num_classes = len(train_ld.dataset.classes)

    # 2. Build EvoCNN with best genome
    best_genome = [3, 1, 3, 2]
    model    = EvoCNN(best_genome, num_classes).to(DEVICE)
    optimizer= optim.Adam(model.parameters(), lr=BASE_LR)
    scaler   = GradScaler()

    # 3. Train for EPOCHS
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc = train_one_epoch(model, train_ld, optimizer, scaler)
        val_acc              = evaluate(model, test_ld)
        print(f"[Evo][Epoch {epoch}/{EPOCHS}] "
              f"Train loss={train_loss:.4f}, acc={train_acc:.4f} — "
              f"Val acc={val_acc:.4f}")

    # 4. Final test accuracy
    test_acc = evaluate(model, test_ld)
    print(f"\n*** EvoCNN Best ({best_genome}) Test Acc: {test_acc:.4f}")

if __name__=="__main__":
    main()