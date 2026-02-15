from FitnessFunctions import FitnessFunction

class TerminationCriterion:
    def __init__(self, fitness: FitnessFunction, n: int, budget: int, jump_k: int | None = None):
        self.fitness = fitness
        self.n = n
        self.budget = budget
        self.jump_k = jump_k

    def is_optimal(self, fx) -> bool:
        if self.jump_k is None:
            return fx >= self.n
        return fx >= self.n + self.jump_k

    def should_stop(self, fx) -> bool:
        if self.fitness.call_count > self.budget:
            return True
        return self.is_optimal(fx)

    def __call__(self, fx) -> bool:
        return self.should_stop(fx)

