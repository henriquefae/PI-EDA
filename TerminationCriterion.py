from FitnessFunctions import FitnessFunction, Jump

class TerminationCriterion:
    def __init__(self, fitness: FitnessFunction, n: int, budget: int):
        self.fitness = fitness
        self.n = n
        self.budget = budget

    def is_optimal(self, fx) -> bool:
        if isinstance(self.fitness, Jump):
            return fx >= self.n + self.fitness.k
        return fx >= self.n

    def should_stop(self, fx) -> bool:
        if self.fitness.call_count > self.budget:
            return True
        return self.is_optimal(fx)

    def __call__(self, fx) -> bool:
        return self.should_stop(fx)

