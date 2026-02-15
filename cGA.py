from FrequencyVector import FrequencyVector
from Individual import *
from FitnessFunctions import FitnessFunction
from TerminationCriterion import TerminationCriterion


def cGA(n: int,k: int, f : FitnessFunction, termination_condition: TerminationCriterion, py_rng, np_rng) -> tuple[FrequencyVector, int]:
    p = FrequencyVector(n)                    # p.vector initialized to 0.5
    t = 0

    while True:
        sampler = IndividualFactory(n, py_rng, np_rng)
        sampler.frequency_vector = p
        x1 = sampler.sample_individual()
        fx1 = f.evaluate(x1)
        if termination_condition(fx1):
            return p, t

        x2 = sampler.sample_individual()
        fx2 = f.evaluate(x2)
        if termination_condition(fx2):
            return p, t

        # choose winner x = argmax f
        if fx2 > fx1:
            x1, x2 = x2, x1

        newP = []
        for i in range(n):
            newP.append(p.vector[i] + (x1.genome[i] - x2.genome[i]) / k)
        p.update_vector(newP)

        t += 1