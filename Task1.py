import random
import numpy as np

#  Implement reasonable data structures for individuals and for frequency vectors, 
# and implement standard bit mutation that operates on your individuals.
# We call each 𝒙 ∈ {0, 1}^n an individual
# We call a vector 𝒑 ∈ [0, 1]^n a frequency vector, where p[i] denotes the independent probability that the sample has a 1 at position i
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
        

class FrequencyVector:
    def __init__(self, n):
        self.n = n
        self.vector = [0.5] * n  # Initialize frequency vector with 0.5 for each position

    def update_vector(self, p):
        """Update the frequency vector that satisfies the border condition"""
        self.vector = [max(1/self.n, min(1- 1/self.n, p[i])) for i in range(self.n)]
        return self.vector