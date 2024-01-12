import csv
import torch
import pandas as pd


class StatisticsReporter:
    def __init__(self):
        self.most_fit_genomes = []
        self.generation_statistics = []

    def best_genomes(self, n):
        """Returns the n fittest genomes ever seen."""
        def key(g):
            return g.fitness

        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        """Returns the fittest genome ever seen"""
        return self.best_genomes(1)[0]

    def highest_score(self):
        """Returns the fitness score for the best genome"""
        return self.best_genome().fitness

    def get_fitness_stat(self, func):
        df = pd.DataFrame().from_dict(self.generation_statistics)
        stats = list(df[f'{func}_fitness'])
        return stats

    def get_fitness_mins(self):
        return self.get_fitness_stat("min")

    def get_fitness_means(self):
        """Get the per-generation mean fitness"""
        return self.get_fitness_stat("mean")

    def get_fitness_stds(self):
        """Get the per-generation standard deviation of the fitness"""
        return self.get_fitness_stat("std")

    def get_fitness_vars(self):
        return self.get_fitness_stat("var")

    def get_fitness_meds(self):
        """Get the per-generation median fitness"""
        return self.get_fitness_stat("med")

    def save(self, crossover_mode='mean', directory='/content/gdrive/MyDrive'):
        self.save_genome_fitness(directory=directory)
        self.save_top_n_genomes(directory=directory, n=3,
                                crossover_mode=crossover_mode)

    def save_genome_fitness(self, directory, delimiter=' ',
                            filename='fitness_history.csv'):
        """Saves the population's best and average fitness"""
        with open(f'{directory}/{filename}', 'w') as f:
            w = csv.writer(f, delimiter=delimiter)

            best_fitness = [g.fitness for g in self.most_fit_genomes]
            avg_fitness = self.get_fitness_means()
            for best, avg in zip(best_fitness, avg_fitness):
                w.writerow([best, avg])

    def save_top_n_genomes(self, directory, n, crossover_mode):
        path = f'{directory}/best_{n}_genomes_{crossover_mode}.pt'
        torch.save(self.best_genomes(n), path)
