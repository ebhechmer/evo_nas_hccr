#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt

# Load histories
with open("checkpoints/baseline_history.json") as f:
    b = json.load(f)
with open("checkpoints/evo_history.json") as f:
    e = json.load(f)

epochs = range(1, len(b["train_acc"])+1)

# 1) Accuracy plot
plt.figure()
plt.plot(epochs, b["train_acc"], label="Baseline Train")
plt.plot(epochs, b["val_acc"],   label="Baseline Val")
plt.plot(epochs, e["train_acc"], label="EvoCNN Train")
plt.plot(epochs, e["val_acc"],   label="EvoCNN Val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Val Accuracy")
plt.legend()
plt.show()

# 2) Loss plot
plt.figure()
plt.plot(epochs, b["train_loss"], label="Baseline Loss")
plt.plot(epochs, e["train_loss"], label="EvoCNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()