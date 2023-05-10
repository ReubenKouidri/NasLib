from .genetics import Genome, FlattenGene, ConvBlock2dGene, LinearGene, MaxPool2dGene, build_layer
import random
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from my_utils.my_utils import get_num_correct
from tqdm import tqdm
from my_utils.ksplit import load_2d_dataset
device = "cpu"

class Population:
    def __init__(self, config):
        self.population = []
        self.generation = 0
        self.config = config

    def initialize(self):
        channels_min = ConvBlock2dGene.allowed_channels[0]
        channels_max = ConvBlock2dGene.allowed_channels[-1]
        grainularity = (channels_max - channels_min) // self.config.algo.population_size
        in_channels = 1

        for i in range(self.config.algo.population_size):
            out_channels = channels_min + i * grainularity
            if out_channels > channels_max:
                print("channels overflow!")
                out_channels = random.choice(ConvBlock2dGene.allowed_channels)

            kernel_size = 10
            default_linear_gene = LinearGene(9_000, 9, False)
            default_maxpool_gene_1 = MaxPool2dGene(3, 3)
            default_maxpool_gene_2 = MaxPool2dGene(3, 3)
            default_flatten_gene = FlattenGene()
            activation = "ReLU"

            conv_gene = ConvBlock2dGene(in_channels, out_channels, kernel_size, activation, bn=True)
            conv_gene_2 = ConvBlock2dGene(out_channels, out_channels, kernel_size, activation, bn=True)

            genome = Genome([conv_gene, default_maxpool_gene_1, conv_gene_2, default_maxpool_gene_2, default_flatten_gene])
            in_neurons = out_channels * genome.outdims ** 2
            lb1 = LinearGene(in_neurons, 9_000, True)
            genome.genes.extend([lb1, default_linear_gene])
            self.population.append(genome)

    def evolve(self):
        for generation in range(self.config.algo.generations):
            self.step()

    def step(self):
        for genome in self.population:
            model = self.build_model(genome)
            print(sum(torch.numel(p) for p in model.parameters() if p.requires_grad))

            fitness = self.evaluate_model(model, self.config.algo.epochs)
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
            model.append(build_layer(gene))

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
        imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
        tgts = tgts.to(device, non_blocking=True)
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
        imgs = imgs.to(device, non_blocking=True, dtype=torch.float32)
        tgts = tgts.to(device, non_blocking=True)
        preds = model(imgs)
        correct += get_num_correct(preds, tgts)
        loss = criterion(preds, tgts)
        total_loss += loss.item()

    # val_acc = correct / len(valloader.dataset)
    val_loss = total_loss / (len(valloader.dataset) / valloader.batch_size)
    return val_loss
