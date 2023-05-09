from .genetics import Genome, FlattenGene, ConvBlock2dGene, LinearGene, build_layer
import random
from trainer import Trainer
import torch


class Population:
    def __init__(self, config):
        self.population = []
        self.generation = 0
        self.config = config
        self.default_db2 = LinearGene(9_000, 9, False)

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

            kernel_size = random.choice(ConvBlock2dGene.allowed_kernel_size)
            activation = "ReLU"

            conv_gene = ConvBlock2dGene(in_channels, out_channels, kernel_size, activation, True)
            fg = FlattenGene()
            genome = Genome([conv_gene, fg])
            in_neurons = out_channels * genome.outdims ** 2
            lb1 = LinearGene(in_neurons, 9_000, True)
            genome.genes.extend([lb1, self.default_db2])
            self.population.append(genome)

    def step(self):
        for genome in self.population:
            model = self.build_model(genome)
            fitness = self.evaluate_model(model)
            genome.fitness = fitness

    def evaluate_model(self, model):
        loss = 0.0
        acc = 0.0
        t = Trainer(self.config, algo_mode=True)
        eval_loss, eval_acc = t(model, epochs=self.config.algo.epochs)
        loss = min(loss, eval_loss)
        acc = min(acc, eval_acc)
        return loss

    @staticmethod
    def build_model(genome: Genome) -> torch.nn.Sequential:
        model = torch.nn.Sequential()
        for gene in genome.genes:
            model.append(build_layer(gene))

        return model

    def __len__(self):
        return len(self.population)



