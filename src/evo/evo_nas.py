import random
import numpy as np
import torch
from deap import base, creator, tools
from torch.utils.data import Subset, DataLoader
from src.data.loader import get_hwdb_loaders
from config import (
    PROXY_EPOCHS, BATCH_SIZE, NUM_WORKERS, DEVICE,
    POP_SIZE, NUM_GEN, MUTATION_RATE, CROSSOVER_RATE,
    SMOKE_SUBSET, SMOKE_EPOCHS
)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 1) Define search space
CONV_CHANNEL_OPTIONS = [16, 32, 64, 128]
FC_OPTIONS           = [128, 256, 512]

# 2) Model factory
class EvoCNN(nn.Module):
    def __init__(self, genome, num_classes):
        super().__init__()
        # genome = [idx1, idx2, idx3, idx_fc]
        c1 = CONV_CHANNEL_OPTIONS[genome[0]]
        c2 = CONV_CHANNEL_OPTIONS[genome[1]]
        c3 = CONV_CHANNEL_OPTIONS[genome[2]]
        fc = FC_OPTIONS[genome[3]]

        self.conv1 = nn.Conv2d(1,   c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1,  c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2,  c3, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(c3 * 8 * 8, fc)
        self.fc2   = nn.Linear(fc, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# 3) Fitness evaluation
def fitness(genome):
    # load full proxies
    train_ld, test_ld = get_hwdb_loaders(BATCH_SIZE, NUM_WORKERS)

    # smoke‐test: use small subset
    sub_N = SMOKE_SUBSET
    train_ds = Subset(train_ld.dataset, list(range(min(sub_N, len(train_ld.dataset)))))
    test_ds  = Subset(test_ld.dataset,  list(range(min(sub_N, len(test_ld.dataset)))))

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # instantiate model
    num_classes = len(train_ld.dataset.dataset.classes) \
        if isinstance(train_ld.dataset, Subset) else len(train_ld.dataset.classes)
    model = EvoCNN(genome, num_classes).to(DEVICE)
    opt   = optim.Adam(model.parameters(), lr=1e-3)

    # proxy (smoke) training
    for _ in range(SMOKE_EPOCHS):
        model.train()
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward()
            opt.step()

    # validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return (correct / total,)

# 4) GA setup
def run_evonas():
    random.seed(42)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # genes: three conv indices, one FC index
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

    pop   = toolbox.population(n=POP_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 5) Evolution loop
    for gen in range(1, NUM_GEN+1):
        print(f"\n=== Generation {gen}/{NUM_GEN} ===")
        # select & clone
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))

        # crossover & mutation
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVER_RATE:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mutant in offspring:
            if random.random() < MUTATION_RATE:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate invalid
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        print(f"Evaluating {len(invalid)} individuals…")
        for idx, ind in enumerate(invalid, 1):
            print(f" → Individual {idx}/{len(invalid)}: genome={ind}")
            ind.fitness.values = toolbox.evaluate(ind)

        # replace population and record stats
        pop[:] = offspring
        record = stats.compile(pop)
        print(f"  Stats: {record}")

    # best
    best = tools.selBest(pop, 1)[0]
    print(f"\n*** Best genome: {best} → Fitness: {best.fitness.values[0]:.4f}")
    return best

if __name__ == "__main__":
    run_evonas()