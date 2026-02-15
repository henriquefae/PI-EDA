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
    
    def copy(self):
        """Return a deep copy of the individual (genome is copied)."""
        return Individual(self.genome.copy(), self.frequency_vector)

    def mutate_individual(self, individual):
        """Apply standard bit mutation to an individual: it flips each bit with a 1 / n mutation rate."""
        # Copy individual
        mutated = self.copy()

        # Sample number of flips
        X = np.random.binomial(n=self.n, p=1/self.n)

        # Random positions
        flip_positions = random.sample(range(self.n), X)

        # Flip bits
        for pos in flip_positions:
            mutated.genome[pos] = 1 - mutated.genome[pos]

        return mutated

class IndividualFactory:
    def __init__(self, n: int):
        self.n = n
        self.frequency_vector = FrequencyVector(self.n)

    def sample_individual(self) -> Individual:
        genome = [1 if random.random() < p else 0 for p in self.frequency_vector.vector]
        return Individual(genome)