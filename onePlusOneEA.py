from FrequencyVector import FrequencyVector
from FitnessFunctions import FitnessFunction
from Individual import *
from TerminationCriterion import TerminationCriterion


def onePlusOneEA(n: int, f : FitnessFunction, termination_condition: TerminationCriterion, py_rng, np_rng) -> tuple[FrequencyVector, int]:
    p = FrequencyVector(n)                         # p.vector initialized to 0.5
    t = 0
    sampler = IndividualFactory(n, py_rng, np_rng)
    sampler.frequency_vector = p
    x = sampler.sample_individual()

    while True:
        y = x.mutate()
        fx = f.evaluate(x)
        fy = f.evaluate(y)
        if fy > fx:
            x, fx = y, fy
        if termination_condition(fx):
            return p, t

        t += 1