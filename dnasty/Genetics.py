import random
from typing import Any, Type, Self, Optional, Iterable, TextIO


def random_mutation(size: int) -> int:
    return random.randrange(-size, size)


class Gene:
    """
    Base class for derived Genes
    Different derived genes encode:
        - Conv layers (size and number of kernels, stride length, padding)
        - MaxPool layers (size of kernel and stride length)
    """

    def __init__(self, id: int, value: int):
        self.id = id
        self.value = value

    def mutate(self): ...


class Chromosome:

    def __init__(self, location: int, genes: list[Gene], chromosome_type: str, id: int) -> None:
        """
        :param location: location in the Genome: used to find final Chromosome and correctly set out layers
        :param genes: a list of genes that make up the chromosome
        :param chromosome_type: type of layer that the chromosome encodes: Union[Conv, Dense]
        :param id: innovation number
        """
        self.location = location
        self.genes = genes
        self._type = chromosome_type
        self.innov_num = id

    @classmethod
    def mutate(cls, genepool: Any, species: Any) -> Type[Self]:
        """
        TODO:
            - keep track of innovation number?
            - mutate genes
            - instead of genepool take the config file
        :param genepool:
        :param species:
        :return Chromosome: new mutated chromosome
        """
        ...

    @classmethod
    def crossover(cls, o: Type[Self], method: str) -> Type[Self]:
        """
        TODO:
            - keep track of innov num
            - cross over genetics

        Cross over genetic information
        Need to keep track of innovation number each time a mutation is performed
        :param o: other chromosome
        :param method: how to mix genetic information
        :return Chromosome: new Chromosome with crossed over genes
        """
        ...

    def __len__(self):
        return len(self.genes)


class Genome:
    """
    A container for all genetic information needed to create a CNN
    """
    def __init__(
            self,
            config_file: TextIO,
            chromosomes: Optional[Iterable[Chromosome]],
            fitness: Optional[float] = 0.0
    ) -> None:
        self.config_file = config_file
        self.fitness = fitness
        self.chromosomes = chromosomes

    def __len__(self):
        return len(self.chromosomes)


    @classmethod
    def crossover(cls, o: Type[Self], gen: int, method: str): ...

    @classmethod
    def mutate(cls): ...
