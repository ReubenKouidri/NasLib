from collections import OrderedDict, namedtuple
from itertools import product


class Controls:
    @staticmethod
    def get_hyperparams():
        hyperparams = OrderedDict(
            train_path=['validation_set'],
            train_ref=['REF.csv'],
            test_path=['validation_set'],
            test_ref=['REF.csv'],
            mutation_rate=[1, 0.01, 0.1, 0.5],
            crossover_mode=['mean', 'random'],
            survival_threshold=[0.1, 0.5],
            keep=[2],
            population_size=[10],
            species=[
                [1, 1], [1, 2], [1, 3],
                [2, 1], [2, 2], [2, 3]
            ]
        )
        return hyperparams


class RunBuilder:
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs
