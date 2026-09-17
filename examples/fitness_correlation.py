"""Random search, then retrain the per-generation best genomes for longer and
plot low-fidelity fitness against the fully-trained score.

Both stages share one ``DataModule`` so they use the same validation split.

    uv run python examples/fitness_correlation.py configs/default.yaml
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dnasty import (
    Config,
    DataModule,
    LowFidelityEstimator,
    RandomSearch,
    Trainer,
    seed_everything,
)


def plot_correlation(fitnesses: list[float], scores: list[float]) -> None:
    df = pd.DataFrame({"Fitness": fitnesses, "Trained Score": scores})
    print("Correlation Matrix:")
    print(df.corr())
    sns.scatterplot(data=df, x="Fitness", y="Trained Score")
    plt.title("Low-fidelity fitness vs. trained score")
    plt.show()


def main(config_path: str) -> None:
    config = Config.from_file(config_path).nas
    seed_everything(config.get("seed", 0))
    datamodule = DataModule.from_config(config)
    trainer = Trainer(config, datamodule)
    strategy = RandomSearch(
        config, estimator=LowFidelityEstimator(config, trainer=trainer)
    )
    strategy.fit()

    best_genomes = sorted(strategy.history, key=lambda g: g.fitness, reverse=True)
    scores = [trainer.fit(g.to_module(), config.train.epochs) for g in best_genomes]
    plot_correlation([g.fitness for g in best_genomes], scores)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
