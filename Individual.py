import random
import numpy as np
from FrequencyVector import FrequencyVector

class Individual:

    def __init__(self, genome : list[int], frequency_vector: FrequencyVector = None, py_rng=None, np_rng=None):
        self.n = len(genome)
        self.genome = genome
        self.py_rng = py_rng
        self.np_rng = np_rng

        if frequency_vector is None:
            self.frequency_vector = FrequencyVector(self.n)
        else:
            self.frequency_vector = frequency_vector
    
    def copy(self):
        """Return a deep copy of the individual (genome is copied)."""
        return Individual(
            self.genome.copy(),
            self.frequency_vector,
            self.py_rng,
            self.np_rng
        )


    def mutate(self):
        """Apply standard bit mutation to an individual: it flips each bit with a 1 / n mutation rate."""
        # Copy individual
        mutated = self.copy()

        # Sample number of flips
        X = self.np_rng.binomial(n=self.n, p=1/self.n)

        # Sample positions to flip
        flip_positions = self.py_rng.sample(range(self.n), X)

        # Flip bits
        for pos in flip_positions:
            mutated.genome[pos] = 1 - mutated.genome[pos]

        return mutated

class IndividualFactory:

    def __init__(self, n: int, py_rng, np_rng):
        self.n = n
        self.frequency_vector = FrequencyVector(self.n)
        self.py_rng = py_rng
        self.np_rng = np_rng

    def sample_individual(self) -> Individual:
        genome = [
            1 if self.py_rng.random() < p else 0
            for p in self.frequency_vector.vector
        ]

        return Individual(
            genome,
            self.frequency_vector,
            self.py_rng,
            self.np_rng
        )
