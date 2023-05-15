import abc
from abc import abstractmethod
import collections.abc
import random
import numpy as np
import inspect
import torch.nn as nn
import dnasty.components as components
import warnings
import copy


__all__ = [
    "LinearGene",
    "ConvBlock2dGene",
    "MaxPool2dGene",
    "FlattenGene",
    "ChannelAttentionGene",
    "SpatialAttentionGene",
    "CBAMGene",
    "Genome"
]


def random_mutation(size: int) -> int:
    return random.randrange(-size, size)


def mag_crossover(g1, g2):
    """return new gene1 where exons take the mean value of both parents"""
    ...


def output_size(input_size, filter_size=1, padding=0, stride=1):
    return 1 + np.floor((input_size - filter_size + 2 * padding) / stride)


def build_layer(gene):
    name = gene.__class__.__name__.replace("Gene", "")
    module = components if hasattr(components, name) else nn
    if not hasattr(module, name):
        raise AttributeError(f"No module {name} found in either torch.nn or dnasty.components")
    sig = inspect.signature(getattr(module, name))

    params = [p for p in gene.exons.keys() if p in sig.parameters]
    kwargs = {p: gene.exons[p] for p in params}
    layer = getattr(module, name)(**kwargs)

    return layer


class GeneBase(abc.ABC):
    """Base class for all Genes to inherit from"""
    @abstractmethod
    def __init__(self, exons):
        if not isinstance(exons, collections.abc.Mapping):
            raise TypeError(f"exons must be a Mapping type.\n"
                            f"exons {exons} of type {type(exons)} were passed.")
        self.exons = dict(exons)

    @abstractmethod
    def mutate(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Mutate function must be implemented!"
        )

    @abstractmethod
    def _validate(self):
        """
        Make sure that any modification to the gene remains within allowed range
        At present this is not to validate the input; those errors LinearGene handled in the __init__ method
        """
        raise NotImplementedError(
            "Must implement abstract method 'validate'!"
        )

    def __getattr__(self, name):
        cls = self.__class__
        if hasattr(cls, name):  # search static attributes
            return getattr(cls, name)
        elif name in self.exons:
            return self.exons[name]
        else:
            raise AttributeError(f"No attribute {name} in {self.__name__ }")

    def __deepcopy__(self, memo=None):
        cls = self.__class__
        exons_copy = copy.deepcopy(self.exons, memo)
        new_obj = cls(**exons_copy)
        memo[id(self)] = new_obj
        return new_obj

    def __len__(self) -> int:
        return len(self.exons)


class LinearGene(GeneBase):
    allowed_features = range(9, 10_000)

    def __init__(self, in_features, out_features, dropout):
        if in_features < LinearGene.allowed_features[0]:
            raise ValueError(f"in_features ({in_features}) must be in the allowed range:"
                             f" >{LinearGene.allowed_features[0]}")
        if out_features not in LinearGene.allowed_features:
            raise ValueError(f"out_features ({out_features}) must be in the allowed range:"
                             f"[{LinearGene.allowed_features[0], LinearGene.allowed_features[-1]}]")
        if not isinstance(dropout, bool):
            raise ValueError(f"dropout {dropout} should be type bool")

        _exons = {"in_features": int(in_features), "out_features": int(out_features), "dropout": dropout}
        super().__init__(_exons)

    def mutate(self) -> None:
        self.exons["dropout"] = not(self.exons["dropout"])
        self.exons["out_features"] = self.exons["out_features"] + random.randrange(-100, 100, 10)
        self._validate()

    def _validate(self):
        if self.exons["out_features"] not in LinearGene.allowed_features:
            if self.exons["out_features"] > LinearGene.allowed_features[-1]:
                warnings.warn(
                    f"out_features {self.exons['out_features']} larger than largest allowed value. "
                    f"Automatically set to largest allowed value {LinearGene.allowed_features[-1]}."
                )
                self.exons["out_features"] = LinearGene.allowed_features[-1]
            else:
                warnings.warn(
                    f"out_features {self.exons['out_features']} smaller than smallest allowed value. "
                    f"Automatically set to smallest allowed value {LinearGene.allowed_features[0]}."
                )
                self.exons["out_features"] = LinearGene.allowed_features[0]


class FlattenGene:
    exons = {"start_dim": 1, "end_dim": -1}


class ConvBlock2dGene(GeneBase):
    allowed_channels = (2**i for i in range(1, 6))
    allowed_kernel_size = range(2, 16, 2)
    allowed_activations = {"ReLU", "tanh"}

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple, activation: str = "ReLU",
                 batch_norm: bool | None = True):
        if activation not in self.__class__.allowed_activations:
            raise ValueError(f"activation {activation} not in allowed set: {self.allowed_activations}")

        exons = {"in_channels": in_channels,
                 "out_channels": out_channels,
                 "kernel_size": kernel_size,
                 "activation": activation,
                 "batch_norm": batch_norm}

        super().__init__(exons)

    def mutate(self, fnc):
        super().mutate(fnc)
    
    def _validate(self): ...


class MaxPool2dGene(GeneBase):
    allowed_values = range(2, 10)

    def __init__(self, kernel_size, stride):
        if kernel_size not in set(MaxPool2dGene.allowed_values):
            raise ValueError(f"kernel_size {kernel_size} must be in {[self.allowed_values[0], self.allowed_values[-1]]}")

        exons = {"kernel_size": kernel_size, "stride": stride}
        super().__init__(exons)

    @staticmethod
    def _mutate():
        if random.random() > 0.5:
            return lambda x: x + 1
        else:
            return lambda x: x -1

    def mutate(self, fnc):
        super().mutate(fnc)

    def _validate(self):
        ...


class ChannelAttentionGene(GeneBase): ...


class SpatialAttentionGene(GeneBase):
    def __init__(self):
        self.exons = {"kernel_size": 5, "e2": 2}

    def mutate(self, fnc):
        super().mutate(fnc)

    def _validate(self): ...


class CBAMGene(GeneBase): ...


class Genome:
    reduce_func = lambda self, d, f, p, s: 1 + (d - f + 2 * p) // s

    @property
    def outdims(self):
        dims = self.image_dims
        for gene in self.genes:
            if isinstance(gene, ConvBlock2dGene):
                dims = self.reduce_func(dims, gene.kernel_size, 0, 1)
            elif isinstance(gene, MaxPool2dGene):
                dims = self.reduce_func(dims, gene.kernel_size, 0, gene.kernel_size)

        return dims

    def _sync_layers(self):
        prev_out_channels = 1
        for gene in self.genes:
            if isinstance(gene, ConvBlock2dGene):
                in_chans = gene.exons["in_channels"]
                gene.exons["in_channels"] = prev_out_channels if in_chans != prev_out_channels else in_chans
                prev_out_channels = gene.exons["out_channels"]

    def __init__(self, genes):
        self.genes = list(genes)
        self.fitness = 0.0
        self.image_dims = 128

    def crossover(self, other, fnc) -> 'Genome': ...

    def __len__(self):
        return len(self.genes)


'''
class Gene:
    """
        - chromosome_type: str
        - tuples encoding each layer
        - layer = CNN: chromosomes = ("conv_ker_0": val, "in_planes_0": val, "out_planes_0": val, "att_ker": val, "r_ratio": val)
        - layer = DL: chromosomes = ("in_layers": val, "out_layers": val, "num_neurons": val)
        - loc1 = loc1 in the genome i.e. list of chromosomes
    """

    def __init__(self, loc1, sequence, gene_type):
        self.species_type = [(2, 2), 2]
        self.sequence = sequence
        self.gene_type = gene_type  # conv gene1 or dense gene1
        self.loc1 = loc1
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

        if self.loc1 == len(self.species_type[0]) + self.species_type[1] - 1:
            self.sequence['out_features'] = 9

    def mix(self, gene2, generation, mean=True):
        """ gene2 from other parent: returns the mean of the chromosomes if 'mean' """
        if mean:
            # print("In MIX before mix: ", self.chromosomes)
            keys = list(self.sequence.keys())
            sequence_1 = np.array(list(self.sequence.values()))
            sequence_2 = np.array(list(gene2.chromosomes.values()))
            new_sequence = sequence_1 + ((math.exp(generation / self.alpha) * (sequence_2 - sequence_1)) // 2).astype(
                int)
            new_sequence = {key: val for key, val in zip(keys, new_sequence)}
            self.sequence = new_sequence
            # print("AFTER mix", self.chromosomes)

        else:
            print("WARNING: gene1 mixing not implemented!")

    def __len__(self):
        return len(self.sequence)


class Genome:
    def __init__(
            self,
            config,
            genes: collections.abc.Iterable,
            innovation_number: int,
            fitness: Optional[float] = 0.0
    ) -> None:
        self.config = config
        self.fitness = fitness
        self.genes = list(genes)
        self.innovation_number = innovation_number

    def crossover(self): ...

    def __len__(self):
        return len(self.genes)


class GenePool:
    def __init__(self, filter_set_conv, kernel_set_conv, kernel_set_att, r_pool, max_pool_set, neuron_set):
        self.outplanes_genepool = filter_set_conv
        self.conv_ker_genepoool = kernel_set_conv
        self.reduction_genepool = r_pool
        self.att_ker_genepool = kernel_set_att
        self.maxpool_genepool = max_pool_set
        self.neuron_genepool = neuron_set


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
            d = output_size(d, self.genes[j].chromosomes['mp_ker'], padding=0,
                            stride=self.genes[j].chromosomes['mp_ker'])
        return d

    def __init__(self, species_type):
        self.species_type = species_type
        self.genes = []
        self.fitness = None

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
            res_locations = res_locations[
                            :res_mixes]  # e.g. chromosomes to be crossed are at loc1 [1] if res_mixes = 1

            dense_mixes = random.choice(np.arange(1, num_dense_genes))  # number of dense chromosomes to be crossed over
            dense_locations = list(np.arange(num_dense_genes))  # locations of the dense chromosomes
            random.shuffle(dense_locations)  # shuffled locations
            dense_locations = dense_locations[:dense_mixes]  # the locations of the dense chromosomes to be crossed
            # dense_locations += num_res_genes  # locations of dense chromosomes are after the res chromosomes

            for loc1 in res_locations:
                self.genes[loc1].mix(genome2.chromosomes[loc1], generation=generation)
                # e.g. res_1.mix(other res_1)
            for loc1 in dense_locations:
                self.genes[num_res_genes + loc1].mix(genome2.chromosomes[num_res_genes + loc1],
                                                         generation=generation)

        elif crossover_mode == 'mean':
            # print("before crossover chromosomes: ")
            # print([gene1.chromosomes for gene1 in self.chromosomes])
            for i in np.arange(self.__len__()):
                self.genes[i].mix(genome2.chromosomes[i], generation=generation)
            # print("after crossover:")
            # print([gene1.chromosomes for gene1 in self.chromosomes])

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
        reproducing_genomes = list(
            self.ranking.iloc[:num_reproducing, 0])  # the set of individuals that are allowed to reproduce

        stats = OrderedDict()
        stats["mean_fitness"] = self.ranking["fitness"].mean()
        stats["med_fitness"] = self.ranking["fitness"].median()
        stats["std_fitness"] = self.ranking["fitness"].std()
        stats["var_fitness"] = self.ranking["fitness"].var()
        stats["min_fitness"] = self.ranking["fitness"].min()
        stats["max_fitness"] = self.ranking["fitness"].max()

        self.history.most_fit_genomes.append(best_k_genomes[0])
        # print(f'best_k: {best_k_genomes}')
        # print(f'best k, 0: {best_k_genomes[0]}')

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
                for gene1 in child.genes:
                    if random.random() < self.run.mutation_rate:
                        gene1.mutate(self.genepool)
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
                for gene1 in child.genes:
                    gene1.mutate(self.genepool)
                if validate_genome(child):
                    extras.append(child)
                    break

        self.population = self.population[:self.__len__() - len(extras)]
        self.population.extend(extras)
        # print(f"POP SIZE: {self.__len__()}")

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
                    gene1 = Gene(loc1=j, sequence=sequence,
                                gene_type=f'RB{species_type[0][j]}')  # should be type=RBx, gene1 should be "chromosome"
                    genome.genes.append(gene1)
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
                        gene1 = Gene(loc1=loc, sequence=sequence, gene_type='dense')
                        genome.genes.append(gene1)

                    self.population.append(genome)
                    break
'''