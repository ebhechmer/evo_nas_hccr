#!/usr/bin/env python3
import os, json
import matplotlib
# 1) Use Agg backend so we can save plots without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 2) Load histories
with open("checkpoints/baseline_history.json") as f:
    baseline = json.load(f)
with open("checkpoints/evo_history.json") as f:
    evo = json.load(f)

# 3) Prepare output dir
os.makedirs("figures", exist_ok=True)

epochs = list(range(1, len(baseline["val_acc"]) + 1))

# 4a) Validation Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["val_acc"], marker='o', label="Baseline Val Acc")
plt.plot(epochs,    evo["val_acc"], marker='s', label="Evo Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Val Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/val_accuracy.png")
plt.close()

# 4b) Training Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["train_acc"], marker='o', label="Baseline Train Acc")
plt.plot(epochs,    evo["train_acc"], marker='s', label="Evo Train Acc")
plt.xlabel("Epoch"); plt.ylabel("Train Accuracy")
plt.title("Training Accuracy per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/train_accuracy.png")
plt.close()

# 4c) Training Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["train_loss"], marker='o', label="Baseline Train Loss")
plt.plot(epochs,    evo["train_loss"], marker='s', label="Evo Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Train Loss")
plt.title("Training Loss per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/train_loss.png")
plt.close()

# 4d) Parameter Count Comparison
params = [baseline["params"], evo["params"]]
labels = ["BaselineCNN", "EvoCNN"]
plt.figure(figsize=(5,4))
plt.bar(labels, params)
plt.ylabel("Parameter Count")
plt.title("Model Size Comparison")
plt.tight_layout()
plt.savefig("figures/param_counts.png")
plt.close()

print("✅ Saved plots to figures/*.png")