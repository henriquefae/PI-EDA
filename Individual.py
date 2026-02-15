import random
import numpy as np
from FrequencyVector import FrequencyVector

class Individual:

    def __init__(self, genome : list[int], frequency_vector: FrequencyVector = None):
        self.n = len(genome)
        self.genome = genome
        if frequency_vector is None:
            self.frequency_vector = FrequencyVector(self.n)
        else:
            self.frequency_vector = frequency_vector

    def mutate_individual(self, individual):
        """Apply standard bit mutation to an individual: it flips each bit with a 1 / n mutation rate."""
        # genrate a random vector of size that follows a binomial distribution with parameters n and 1/n
        X = np.random.binomial(n=self.n, p=1/self.n)
        # randomly select X positions to flip
        flip_positions = random.sample(range(self.n), X)

        mutated_individual = individual.copy()
        # flip the selected positions
        for pos in flip_positions:
            mutated_individual.genome[pos] = 1 - mutated_individual.genome[pos]
        return mutated_individual

class IndividualFactory:
    def __init__(self, n: int):
        self.n = n
        self.frequency_vector = FrequencyVector(self.n)

    def sample_individual(self) -> Individual:
        genome = [1 if random.random() < p else 0 for p in self.frequency_vector.vector]
        return Individual(genome)