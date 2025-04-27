#!/usr/bin/env python3
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from deap import base, creator, tools

from config import (
    DEVICE, BATCH_SIZE, NUM_WORKERS,
    SMOKE_SUBSET, SMOKE_EPOCHS,
    POP_SIZE, NUM_GEN, MUTATION_RATE, CROSSOVER_RATE
)
from src.data.loader import get_hwdb_loaders

# Speed up conv kernels for fixed input size
cudnn.benchmark = True

# Preload the full datasets once
train_loader_full, test_loader_full = get_hwdb_loaders()
train_dataset_full = train_loader_full.dataset
test_dataset_full  = test_loader_full.dataset

# Search‐space definitions
CONV_CHANNEL_OPTIONS = [16, 32, 64, 128]
FC_OPTIONS           = [128, 256, 512]

class EvoCNN(nn.Module):
    def __init__(self, genome, num_classes):
        super().__init__()
        c1, c2, c3 = [CONV_CHANNEL_OPTIONS[i] for i in genome[:3]]
        fc_units   = FC_OPTIONS[genome[3]]

        self.conv1 = nn.Conv2d(1,   c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1,  c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2,  c3, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(c3 * 8 * 8, fc_units)
        self.fc2   = nn.Linear(fc_units, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def fitness(genome):
    # Build small proxy DataLoaders
    sub_N = SMOKE_SUBSET
    epochs = SMOKE_EPOCHS

    train_ds = Subset(train_dataset_full, list(range(min(sub_N, len(train_dataset_full)))))
    test_ds  = Subset(test_dataset_full,  list(range(min(sub_N, len(test_dataset_full)))))

    train_ld = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_ld  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Instantiate model, optimizer, scaler
    num_classes = len(train_dataset_full.classes)
    model = EvoCNN(genome, num_classes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    # Proxy training loop
    for _ in range(epochs):
        model.train()
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                loss = F.cross_entropy(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # Proxy validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            with autocast():
                preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

    return (correct / total,)

def run_evonas():
    random.seed(42)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("conv_gene", random.randrange, len(CONV_CHANNEL_OPTIONS))
    toolbox.register("fc_gene",   random.randrange, len(FC_OPTIONS))
    toolbox.register(
        "individual", tools.initCycle, creator.Individual,
        (toolbox.conv_gene, toolbox.conv_gene, toolbox.conv_gene, toolbox.fc_gene),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate",    tools.cxOnePoint)
    toolbox.register("mutate",  tools.mutUniformInt,
                     low=[0,0,0,0],
                     up=[len(CONV_CHANNEL_OPTIONS)-1]*3 + [len(FC_OPTIONS)-1],
                     indpb=MUTATION_RATE)
    toolbox.register("select",  tools.selTournament, tournsize=3)
    toolbox.register("evaluate", fitness)

    pop = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    for gen in range(1, NUM_GEN+1):
        print(f"\n=== Generation {gen}/{NUM_GEN} ===")
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate new individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluating {len(invalid)} individuals…")
        for idx, ind in enumerate(invalid, 1):
            print(f" → Individual {idx}/{len(invalid)}: genome={ind}")
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        record = stats.compile(pop)
        print(f"  Stats: {record}")

    best = tools.selBest(pop, 1)[0]
    print(f"\n*** Best genome: {best} → Fitness: {best.fitness.values[0]:.4f}")
    return best

if __name__ == "__main__":
    run_evonas()