import torch
from torch import nn
import math
import numpy as np
import random
import pandas as pd
from collections import OrderedDict
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import csv
from ArrhythmiaDataset2D import ArrhythmiaDataset
from uga2_runs import RunBuilder, Controls
from torch.utils.data import random_split
from typing import Union, List
import time
import argparse
from the_creator import create_model


def output_size(input_size, filter_size=1, padding=0, stride=1):
    return 1 + np.floor((input_size - filter_size + 2 * padding) / stride)


@torch.no_grad()
def get_num_correct(preds, labels) -> int:
    return preds.argmax(dim=1).eq(labels).sum().item()


def create_features(num_layers, pool):
    features = []
    for k in np.arange(num_layers):
        feature = random.choice(pool)
        features.append(feature)
    features = sorted(features, reverse=True)
    return features


def random_mutation(size):
    return random.randrange(-size, size)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def validate_genome(genome, d=128):
    # calculate output size from network
    for j in range(len(genome.id[0])):  # for every RB
        for k in range(genome.id[0][j]):  # e.g. 2 sets of conv sequences per RB2
            d = output_size(d, genome.chromosomes[j].chromosomes[f'conv_ker_{k}'])
        d = output_size(d, genome.chromosomes[j].chromosomes['mp_ker'], padding=0, stride=genome.chromosomes[j].chromosomes['mp_ker'])

    if d > 1:
        return True
    return False


def load_data():
    trainset = ArrhythmiaDataset(args.train_dir, reference_file_csv=args.train_ref,
                                 leads=2, normalize=False,
                                 smoothen=False, wavelet='mexh'
                                 )

    testset = ArrhythmiaDataset(args.test_dir, reference_file_csv=args.test_ref,
                                leads=2, normalize=False,
                                smoothen=False, wavelet='mexh'
                                )

    return trainset, testset


def train(model, optimizer, criterion, trainloader, device):
    total_loss = 0
    epoch_steps = 0
    model.train()
    for images, labels in trainloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        loss = criterion(pred, labels)
        loss.backward()

        total_loss += loss.item()
        epoch_steps += 1

        optimizer.step()

    return total_loss / epoch_steps


@torch.no_grad()
def evaluate(model, criterion, valloader, device):
    model.validate(,
    total_loss = 0.0
    epoch_steps = 0
    total = 0
    correct = 0

    for inputs, labels in valloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += get_num_correct(predicted, labels)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        epoch_steps += 1

    val_acc = correct / total
    val_loss = total_loss / epoch_steps
    return val_loss, val_acc


@torch.no_grad()
def test(model, testloader, device):
    correct = 0
    total = 0
    model.validate(,
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += get_num_correct(predicted, labels)

    return correct


def assess_genome(genome):
    model = create_model(genome)  # express chromosomes
    trainset, testset = load_data()

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )
    # e.g. 300 -> 240|60

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=40,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=10,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=50,
        shuffle=False,
        num_workers=4
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        train_loss = train(model, optimizer, criterion, trainloader, device)
        # eval_loss, eval_acc = evaluate(model, criterion, valloader, device)

    fitness = test(model, valloader, device)
    return fitness


class GenePool:
    outplanes = [range(2, 128, 4)]  # num filters in layer
    conv_ker = [range(2, 32)]  # size of conv filter
    r_ratio = [range(1, 10)]  # reduction r_ratio
    att_ker = [range(1, 10)]  # size of att_ker filter
    maxpool = [range(1, 10)]  # size of maxpool kernel
    neurons = [range(1, 10_000)]  # number of out features


class Gene:
    """
        - chromosome_type: str
        - tuples encoding each layer
        - layer = CNN: chromosomes = ("conv_ker_0": val, "in_planes_0": val, "out_planes_0": val, "att_ker": val, "r_ratio": val)
        - layer = DL: chromosomes = ("in_layers": val, "out_layers": val, "num_neurons": val)
        - loc1 = loc1 in the genome i.e. list of chromosomes
    """

    def __init__(self, location, sequence, gene_type):
        self.species_type = [(2, 2), 2]
        self.sequence = sequence
        self.gene_type = gene_type  # conv gene1 or dense gene1
        self.location = location
        self.alpha = 100

    def _mutate_x(self, x, genepool_set, size):
        mutation = random_mutation(size)
        while self.sequence[x] + mutation not in genepool_set:
            mutation = random_mutation(size)
        else:
            self.sequence[x] += mutation

    def mutate(self, genepool):
        if self.gene_type in ['RB1', 'RB2']:
            for i in range(int(self.gene_type[-1])):
                self._mutate_x(f'out_planes_{i}', genepool.outplanes_genepool, 5)
                self._mutate_x(f'conv_ker_{i}', genepool.conv_ker_genepoool, 1)

            self._mutate_x('att_ker', genepool.att_ker_genepool, 1)
            self._mutate_x('r_ratio', genepool.reduction_genepool, 1)

        elif self.gene_type == 'dense':
            self._mutate_x('out_features', np.arange(9, 100000), 100)

        if self.location == len(self.species_type[0]) + self.species_type[1] - 1:
            self.sequence['out_features'] = 9

    def mix(self, gene2, generation, mean=True):
        """ gene2 from other parent: returns the mean of the chromosomes if 'mean' """
        if mean:
            #print("In MIX before mix: ", self.chromosomes)
            keys = list(self.sequence.keys())
            sequence_1 = np.array(list(self.sequence.values()))
            sequence_2 = np.array(list(gene2.chromosomes.values()))
            new_sequence = sequence_1 + ((math.exp(generation / self.alpha) * (sequence_2 - sequence_1)) // 2).astype(
                int)
            new_sequence = {key: val for key, val in zip(keys, new_sequence)}
            self.sequence = new_sequence
            #print("AFTER mix", self.chromosomes)

        else:
            print("WARNING: gene1 mixing not implemented!")

    def __len__(self):
        return len(self.sequence)


class Genome:
    """
        - species_id/type is a list of the number of ResBlocks and dense layers: [ResBlocks, dense_layers]
        - chromosomes contains the chromosomes of 'Chromosome' instances
    """

    @property
    def id(self):
        return self.species_type

    @property
    def output_size(self):
        d = 128
        for j in range(len(self.id[0])):  # for every RB
            for k in range(self.id[0][j]):  # e.g. 2 sets of conv sequences per RB2
                d = output_size(d, self.genes[j].chromosomes[f'conv_ker_{k}'])
            d = output_size(d, self.genes[j].chromosomes['mp_ker'], padding=0, stride=self.genes[j].chromosomes['mp_ker'])
        return d

    def crossover(self, parent2, generation, crossover_mode):
        """
            :returns: child
            - genome.chromosomes = e.g. [res1, res2, dense1, dense2]
            - res.chromosomes = (conv_ker, in_planes_0, out_planes, att_ker, r_ratio, mp_ker)
            - dense.chromosomes = (in_features, out_features)
        """
        genome2 = parent2
        # print("IN CROSSOVER")
        if crossover_mode == 'random':
            # randomly selecting the chromosomes to be crossed - not all chromosomes are crossed
            num_res_genes = len(self.species_type[0])  # total number of res chromosomes
            num_dense_genes = self.species_type[1]  # total number of dense chromosomes
            res_mixes = random.choice(
                np.arange(1, num_res_genes))  # randomly choose the number of res chromosomes to be crossed (at least 1)
            res_locations = list(np.arange(num_res_genes))  # [0, 1]
            random.shuffle(res_locations)  # e.g. [1, 0]
            res_locations = res_locations[:res_mixes]  # e.g. chromosomes to be crossed are at loc1 [1] if res_mixes = 1

            dense_mixes = random.choice(np.arange(1, num_dense_genes))  # number of dense chromosomes to be crossed over
            dense_locations = list(np.arange(num_dense_genes))  # locations of the dense chromosomes
            random.shuffle(dense_locations)  # shuffled locations
            dense_locations = dense_locations[:dense_mixes]  # the locations of the dense chromosomes to be crossed
            # dense_locations += num_res_genes  # locations of dense chromosomes are after the res chromosomes

            for location in res_locations:
                self.genes[location].mix(genome2.chromosomes[location], generation=generation)
                # e.g. res_1.mix(other res_1)
            for location in dense_locations:
                self.genes[num_res_genes + location].mix(genome2.chromosomes[num_res_genes + location],
                                                         generation=generation)

        elif crossover_mode == 'mean':
            # print("before crossover chromosomes: ")
            # print([gene1.chromosomes for gene1 in self.chromosomes])
            for i in np.arange(self.__len__()):
                self.genes[i].mix(genome2.chromosomes[i], generation=generation)
            # print("after crossover:")
            # print([gene1.chromosomes for gene1 in self.chromosomes])

    def __init__(self, species_type):
        self.species_type = species_type
        self.genes = []
        self.fitness = None

    def __len__(self):
        return len(self.genes)


def create_rb_sequence(variant, genepool):
    """
    variant: either '1' or '2'
    """
    sequence = OrderedDict()
    for k in np.arange(variant):  # e.g. 2 sets of conv sequences per RB2
        sequence[f'in_planes_{k}'] = random.choice(genepool.outplanes_genepool)
        sequence[f'out_planes_{k}'] = random.choice(genepool.outplanes_genepool)
        sequence[f'conv_ker_{k}'] = random.choice(genepool.conv_ker_genepoool)
    sequence['att_ker'] = random.choice(genepool.att_ker_genepool)
    sequence['r_ratio'] = random.choice(genepool.reduction_genepool)
    sequence['mp_ker'] = random.choice(genepool.maxpool_genepool)
    return sequence


class Population:
    """
        - keep: number of models to be cloned
        - survival_threshold: proportion of models that will survive and reproduce
        - mutation_rate: rate of mutation
        - genepool: allowed feature values
    """

    def __init__(self, genepool, run):
        self.genepool = genepool
        self.run = run
        self.history = StatisticsReporter()
        self.ranking = pd.DataFrame()
        self.results = []
        self.population = []
        self.generation = 0

    def __len__(self):
        return len(self.population)

    def zero(self):
        self.results = []
        self.generation += 1
        self.ranking = pd.DataFrame()

    def save(self, basepath):
        self.history.save(basepath)

    def assess_genomes(self):
        for genome in self.population:
            genome.fitness = assess_genome(genome)
            d = OrderedDict()
            d["genome"] = genome
            d["fitness"] = genome.fitness
            self.results.append(d)

    def fuck(self):
        self.ranking = pd.DataFrame().from_dict(
            self.results, orient='columns').sort_values("fitness", axis=0, ascending=False)
        best_k_genomes = list(self.ranking.iloc[:args.keep, 0])
        num_reproducing = (np.floor(self.run.survival_threshold * args.pop_size)).astype(int)
        reproducing_genomes = list(self.ranking.iloc[:num_reproducing, 0])  # the set of individuals that are allowed to reproduce

        stats = OrderedDict()
        stats["mean_fitness"] = self.ranking["fitness"].mean()
        stats["med_fitness"] = self.ranking["fitness"].median()
        stats["std_fitness"] = self.ranking["fitness"].std()
        stats["var_fitness"] = self.ranking["fitness"].var()
        stats["min_fitness"] = self.ranking["fitness"].min()
        stats["max_fitness"] = self.ranking["fitness"].max()

        self.history.most_fit_genomes.append(best_k_genomes[0])
        #print(f'best_k: {best_k_genomes}')
        #print(f'best k, 0: {best_k_genomes[0]}')

        self.history.generation_statistics.append(stats)

        children = []

        """j = 0
        j_max = self.__len__() - self.configs.keep
        while j < j_max:
            parent_1 = random.choice(reproducing_genomes)
            parent_2 = random.choice(reproducing_genomes)  # randomly select individuals for breeding
            child = Genome(parent_1.id)
            child.chromosomes = parent_1.chromosomes
            child.crossover(parent_2, generation=self.generation, crossover_mode=self.configs.crossover_mode)
            for gene1 in child.chromosomes:
                if random.random() < self.configs.mutation_rate:
                    gene1.mutate(self.genepool)
            if validate_genome(child, d=128):
                children.append(child)
                j += 1"""

        for k in range(self.__len__() - args.keep):  # create (pop_size - keep) new genomes
            while True:
                parent_1 = random.choice(reproducing_genomes)
                parent_2 = random.choice(reproducing_genomes)  # randomly select individuals for breeding
                child = Genome(parent_1.id)
                child.genes = parent_1.chromosomes
                child.crossover(parent_2, generation=self.generation, crossover_mode=self.run.crossover_mode)
                for gene in child.genes:
                    if random.random() < self.run.mutation_rate:
                        gene.mutate(self.genepool)
                if validate_genome(child, d=128):
                    children.append(child)
                    break

        self.population[args.keep:] = children  # replace the bottom n-k genomes | n=pop_size, k=keep
        self.population[:args.keep] = best_k_genomes
        # create k more models by mutating the top-k models
        # these will be added to the new pop as well as the top-k
        extras = []
        for genome in best_k_genomes:
            while True:
                child = Genome(species_type=genome.id)
                child.genes = genome.chromosomes
                for gene in child.genes:
                    gene.mutate(self.genepool)
                if validate_genome(child):
                    extras.append(child)
                    break

        self.population = self.population[:self.__len__() - len(extras)]
        self.population.extend(extras)
        #print(f"POP SIZE: {self.__len__()}")

    def initiate_population(self, species_type: List) -> None:
        """
            :param species_type: e.g. [(ResBlocks, num_conv layers per RB),  dense_layers] = [(2,2), 2].

            Notes:
                A gene1 is created for each ResBlock and dense layer.
                Dense layer chromosomes just specify the number of neurons,
                whereas ResBlock chromosomes specify more...
        """
        # [(1, 2, 2), 2]
        for i in np.arange(args.pop_size):  # create a population of pop_size genomes, e.g. 100 genomes
            while True:
                genome = Genome(species_type=species_type)  # initiate a genome object
                for j in range(len(species_type[0])):  # for every RB
                    sequence = create_rb_sequence(variant=species_type[0][j], genepool=self.genepool)
                    gene = Gene(location=j, sequence=sequence,
                                gene_type=f'RB{species_type[0][j]}')  # should be type=RBx, gene1 should be "chromosome"
                    genome.genes.append(gene)
                if genome.output_size > 2:
                    start = len(species_type[0])
                    features = create_features(num_layers=species_type[1], pool=self.genepool.neuron_genepool)
                    for i in np.arange(1, len(features)):
                        while features[i] > (features[i - 1] // 2).astype(int):
                            features = create_features(num_layers=species_type[1], pool=self.genepool.neuron_genepool)

                    for loc, feature in enumerate(features):
                        loc += start
                        sequence = OrderedDict()
                        sequence["in_features"] = feature
                        sequence["out_features"] = feature
                        gene = Gene(location=loc, sequence=sequence, gene_type='dense')
                        genome.genes.append(gene)

                    self.population.append(genome)
                    break
