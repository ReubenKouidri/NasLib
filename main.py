from dnasty.my_utils.config import Config
from dnasty.search_strategies import RandomSearch
from dnasty.defaults import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

JSON_FILE = "dnasty/my_utils/config.json"


def plot_correlation(fitnesses: list, scores: list):
    """
    Args:
        fitnesses (list): estimated validation accuracies (before training)
        scores (list): validation accuracies after training
    """
    # Create DataFrame
    data = {'Fitness': fitnesses, 'Trained Score': scores}
    df = pd.DataFrame(data)

    # Calculate correlation
    correlation = df.corr()
    print("Correlation Matrix:")
    print(correlation)

    # Plot correlation heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title("Fitness vs. Trained Score")
    plt.show()

    # Scatter plot
    sns.scatterplot(data=df, x='Fitness', y='Trained Score')
    plt.title("Fitness vs. Trained Score")
    plt.show()


def extract_fitness_and_scores(genome_rank: dict):
    """Extract fitness and corresponding scores from ranked genomes."""
    f, s = zip(
        *[(genome.fitness, score) for genome, score in genome_rank.items()])
    return list(f), list(s)


if __name__ == '__main__':
    config = Config.from_file(JSON_FILE).nas
    trainer = Trainer(config)
    strategy = RandomSearch(config)
    strategy.fit()
    best_genomes = sorted(strategy.history, key=lambda x: x.fitness,
                          reverse=True)
    rank = {genome: trainer.fit(genome.to_module(), config.train.epochs)
            for genome in best_genomes}

    fitness, scores = extract_fitness_and_scores(rank)
    plot_correlation(fitness, scores)
