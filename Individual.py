import random
import numpy as np
from FrequencyVector import FrequencyVector

class Individual:
    def __init__(self, n):
        self.n = n
        self.frequency_vector = FrequencyVector(n) # Initialize frequency vector with 0.5 for each position

    def sample_individual(self):
        """Sample an individual based on the current frequency vector."""
        return [1 if random.random() < p else 0 for p in self.frequency_vector.vector]

    def mutate_individual(self, individual):
        """Apply standard bit mutation to an individual: it flips each bit with a 1 / n mutation rate."""
        # genrate a random vector of size that follows a binomial distribution with parameters n and 1/n
        X = np.random.binomial(n=self.n, p=1/self.n)
        # randomly select X positions to flip
        flip_positions = random.sample(range(self.n), X)
        # copy the individual to avoid modifying the original
        mutated_individual = individual.copy()
        # flip the selected positions
        for pos in flip_positions:
            mutated_individual[pos] = 1 - mutated_individual[pos]  # Flip the bit (0 to 1 or 1 to 0)
        return mutated_individual