import csv
import torch
import pandas as pd


class StatisticsReporter:
    """
        - Gathers (via the reporting interface) and provides (to callers and/or a file)
          the most-fit genomes and information on genome/species fitness and species sizes.
        - Does NOT log models due to memory, and the fact that models can be recreated from the genomes
    """

    def __init__(self):
        self.most_fit_genomes = []  # list containing the fittest genomes from each generation
        self.generation_statistics = []  # list of dicts containing: mean, med, std, var per generation

    def best_genomes(self, n):
        """Returns the n most fit genomes ever seen."""  # *******GOOD
        def key(g):
            return g.fitness
        return sorted(self.most_fit_genomes, key=key, reverse=True)[:n]

    def best_genome(self):
        """Returns the most fit genome ever seen."""  # ********GOOD
        return self.best_genomes(1)[0]

    def highest_score(self):
        """Returns the fitness score for the best genome"""  # ******GOOD
        return self.best_genome().fitness

    def get_fitness_stat(self, func):  # ***** GOOD
        df = pd.DataFrame().from_dict(self.generation_statistics)  # list of dicts, so use from_dict
        stats = list(df[f'{func}_fitness'])
        return stats
    
    def get_fitness_mins(self):
        return self.get_fitness_stat("min")

    def get_fitness_means(self):
        """Get the per-generation mean fitness."""
        return self.get_fitness_stat("mean")

    def get_fitness_stds(self):
        """Get the per-generation standard deviation of the fitness."""
        return self.get_fitness_stat("std")

    def get_fitness_vars(self):
        return self.get_fitness_stat("var")

    def get_fitness_meds(self):
        """Get the per-generation median fitness."""
        return self.get_fitness_stat("med")

    def save(self, crossover_mode='mean', directory='/content/gdrive/MyDrive'):
        self.save_genome_fitness(directory=directory)
        self.save_top_n_genomes(directory=directory, n=3, crossover_mode=crossover_mode)

    def save_genome_fitness(self, directory, delimiter=' ', filename='fitness_history.csv'):
        """ Saves the population's best and average fitness. """
        with open(f'{directory}/{filename}', 'w') as f:
            w = csv.writer(f, delimiter=delimiter)

            best_fitness = [g.fitness for g in self.most_fit_genomes]
            avg_fitness = self.get_fitness_means()
            for best, avg in zip(best_fitness, avg_fitness):
                w.writerow([best, avg])

    def save_top_n_genomes(self, directory, n, crossover_mode, filepath):
        """
        Make sure the path contains info to identify mutation mode
        filename = best_n_genomes
        crossover_mode = 'mean' or 'rand'
        """
        path = f'{directory}{filepath}/best_{n}_genomes_xxx_{crossover_mode}.pt'
        torch.save(self.best_genomes(n), path)
