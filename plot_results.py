#!/usr/bin/env python3
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1) Load histories
with open("checkpoints/baseline_history.json") as f:
    baseline = json.load(f)
with open("checkpoints/evo_history.json") as f:
    evo = json.load(f)

# 2) Hard-coded parameter counts
baseline_params = 3_199_062
evo_params      = 6_284_022

# 3) Prepare output dir
os.makedirs("figures", exist_ok=True)

# 4) Epoch list
epochs = list(range(1, len(baseline["val_acc"]) + 1))

# 5a) Validation Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["val_acc"], marker='o', label="Baseline Val Acc")
plt.plot(epochs, evo["val_acc"],    marker='s', label="Evo Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/val_accuracy.png")
plt.close()

# 5b) Training Accuracy
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["train_acc"], marker='o', label="Baseline Train Acc")
plt.plot(epochs, evo["train_acc"],    marker='s', label="Evo Train Acc")
plt.xlabel("Epoch"); plt.ylabel("Training Accuracy")
plt.title("Training Accuracy per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/train_accuracy.png")
plt.close()

# 5c) Training Loss
plt.figure(figsize=(6,4))
plt.plot(epochs, baseline["train_loss"], marker='o', label="Baseline Train Loss")
plt.plot(epochs, evo["train_loss"],    marker='s', label="Evo Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Training Loss")
plt.title("Training Loss per Epoch")
plt.legend(); plt.tight_layout()
plt.savefig("figures/train_loss.png")
plt.close()

# 5d) Parameter Count Comparison
plt.figure(figsize=(5,4))
plt.bar(["BaselineCNN","EvoCNN"], [baseline_params, evo_params])
plt.ylabel("Parameter Count")
plt.title("Model Size Comparison")
plt.tight_layout()
plt.savefig("figures/param_counts.png")
plt.close()

print("✅ Saved figures in the figures/ directory:")
for fn in ["val_accuracy.png","train_accuracy.png","train_loss.png","param_counts.png"]:
    print("   figures/" + fn)