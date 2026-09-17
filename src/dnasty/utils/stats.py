from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import torch


class StatisticsReporter:
    """Collects per-generation fitness statistics and the best genomes."""

    def __init__(self) -> None:
        self.most_fit_genomes: list = []
        self.generation_statistics: list[dict] = []

    def best_genomes(self, n: int) -> list:
        """Returns the n fittest genomes ever seen."""
        return sorted(self.most_fit_genomes, key=lambda g: g.fitness, reverse=True)[:n]

    @property
    def max_fitness(self):
        """Returns the best genome seen."""
        return self.best_genomes(1)[0]

    @property
    def min_fitness(self) -> list[float]:
        return self._get_fitness_stat("min")

    @property
    def mean_fitness(self) -> list[float]:
        return self._get_fitness_stat("mean")

    def _get_fitness_stat(self, stat: str) -> list[float]:
        df = pd.DataFrame(self.generation_statistics)
        return list(df[f"{stat}_fitness"])

    def get_fitness_stds(self) -> list[float]:
        return self._get_fitness_stat("std")

    def get_fitness_vars(self) -> list[float]:
        return self._get_fitness_stat("var")

    def get_fitness_meds(self) -> list[float]:
        return self._get_fitness_stat("med")

    def save(self, directory: str | Path, tag: str = "run") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_genome_fitness(directory)
        self.save_top_n_genomes(directory, n=3, tag=tag)

    def save_genome_fitness(
        self,
        directory: str | Path,
        delimiter: str = " ",
        filename: str = "fitness_history.csv",
    ) -> None:
        """Saves the population's best and average fitness per generation."""
        with (Path(directory) / filename).open("w", newline="") as f:
            w = csv.writer(f, delimiter=delimiter)
            best_fitness = [g.fitness for g in self.most_fit_genomes]
            for best, avg in zip(best_fitness, self.mean_fitness, strict=False):
                w.writerow([best, avg])

    def save_top_n_genomes(self, directory: str | Path, n: int, tag: str) -> None:
        torch.save(self.best_genomes(n), Path(directory) / f"best_{n}_genomes_{tag}.pt")
