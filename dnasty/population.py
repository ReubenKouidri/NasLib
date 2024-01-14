from dnasty.genetics import *
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from my_utils.my_utils import get_num_correct
from tqdm import tqdm
from dnasty.my_utils.ksplit import load_2d_dataset
import copy
from dnasty.my_utils.config import Config
import json
import os

""" 
OLD CODE!
Will be completely replaced by a new interface! Kept for reference only. 
"""

local = os.getcwd()
config_path = os.path.join(local, "dnasty/my_utils/nas_config.json")


def load_json(json_file: str) -> dict:
    with open(json_file) as fp:
        return json.load(fp)

    
config = Config(load_json("nas_config.json"))


DEFAULT_CONV_KER = 10
DEFAULT_MAXPOOL_KER = 3
DEFAULT_ACTIVATION = "ReLU"
DEFAULT_CONV_SEARCH_ORDER = ["kernel_size", "out_channels", "activation"]


# TODO:
#   - write Population class then search ONLY for the best conv block
#   - then modify to iteratively search CBAM components


# TODO:
#   - init pop:
#       * cover widest range, or increasing from min? Is the latter an EA?
#   - train, eval
#   - select, crossover and mutate:
#       * add special methods for crossover involving simple arithmetic e.g. __add__ and __div__ for (g1+g2)/2
#   - save best Genome + eval metrics
#   - repeat for N generations


class Population:
    def __init__(self, config: Config):
        self.best_conv_genes = {}
        self.target_gene = ConvBlock2dGene
        self.target_exon = config.default_search_order[0]  # kernel_size exon
        self.population = []
        self.generation = 0
        self.config = config

    def search(self):
        """method that searches for the best specified exon in a certain gene"""
        target_exon = "kernel_size"
        save_id = f"targets<gene<{self.target_gene}>_exon<{target_exon}>>"
        allowed_ker_sizes = ConvBlock2dGene.allowed_kernel_size
        default_exons = {"in_channels": 1, "out_channels": 16, "activation": "ReLU"}

        for i in range(len(allowed_ker_sizes)):  # e.g. range(5)
            kernel_size = allowed_ker_sizes[i]
            default_exons["kernel_size"] = kernel_size
            mp_gene = MaxPool2dGene(3, 3)
            flatten_gene = FlattenGene()
            NUM_CONV = 1
            NUM_STACKS = 3
            genes = [ConvBlock2dGene(**default_exons) for c in NUM_CONV]
            genes.append(MaxPool2dGene(3, 3))
            genes = copy.deepcopy(genes) * 3  # [c, mp, c, mp, c, mp]

            for i in range(len(genes)):
                if i > 0:
                    genes[i].in_channels = genes[i - 1].out_channels

            genes.extend([mp_gene, flatten_gene])
            genome = Genome(genes)
            in_neurons = int(default_exons.out_channels * genome.outdims ** 2)
            lb1 = LinearBlockGene(in_neurons, 9_000, True)
            output_gene = LinearBlockGene(9_000, 9, False)
            genome.genes.extend([lb1, output_gene])
            self.population.append(genome)

        self.step()

    def evolve(self):
        for generation in range(self.config.generations):
            self.step()

    def step(self):
        for genome in self.population:
            model = self.build_model(genome)
            print(f"Num trainable params: {sum(torch.numel(p) for p in model.parameters() if p.requires_grad)}\n")

            fitness = self.evaluate_model(model, self.config.epochs)
            print(fitness)
            genome.fitness = fitness
        self.population = sorted(self.population, key=lambda g: g.fitness, reverse=False)
        print([g.fitness for g in self.population])

    def evaluate_model(self, model, epochs):
        dataset = load_2d_dataset(self.config.data.data_path, self.config.data.reference_path)
        test_abs = int(len(dataset) * 0.8)
        trainset, valset = random_split(
            dataset, [test_abs, len(dataset) - test_abs]
        )
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=40,
            shuffle=True
        )
        valloader = torch.utils.data.DataLoader(
            valset,
            batch_size=10,
            shuffle=True,
        )
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9, nesterov=True)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            train(model, optimizer, criterion, trainloader)
        fitness = validate(model, valloader, criterion)

        return fitness

    @staticmethod
    def build_model(genome: Genome) -> torch.nn.Sequential:
        model = torch.nn.Sequential()
        for gene in genome.genes:
            model.append(_build_layer(gene))

        return model

    def __len__(self):
        return len(self.population)


def train(model, optimizer, criterion, trainloader):
    print("training")
    model.train()
    total_loss = 0.0
    correct = 0
    trainloader.dataset.dataset.test = False
    for imgs, tgts in tqdm(trainloader):
        imgs = imgs.to(config.device, non_blocking=True, dtype=torch.float32)
        tgts = tgts.to(config.device, non_blocking=True)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, tgts)
        loss.backward()
        correct += get_num_correct(preds, tgts)
        total_loss += loss.item()
        optimizer.step()

    # train_loss = total_loss / (len(trainloader.dataset) / trainloader.batch_size)
    # train_acc = correct / (len(trainloader.dataset))
    #
    # return train_loss, train_acc


@torch.inference_mode()
def validate(model, valloader, criterion) -> tuple[float, float]:
    print("validating")
    model.eval()
    total_loss = 0.0
    correct = 0
    valloader.dataset.dataset.test = False
    for imgs, tgts in tqdm(valloader):
        imgs = imgs.to(config.device, non_blocking=True, dtype=torch.float32)
        tgts = tgts.to(config.device, non_blocking=True)
        preds = model(imgs)
        correct += get_num_correct(preds, tgts)
        loss = criterion(preds, tgts)
        total_loss += loss.item()

    # val_acc = correct / len(valloader.dataset)
    val_loss = total_loss / (len(valloader.dataset) / valloader.batch_size)
    return val_loss
