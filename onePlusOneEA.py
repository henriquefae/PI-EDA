from FrequencyVector import FrequencyVector
from FitnessFunctions import FitnessFunction
from Individual import *


def onePlusOneEA(n: int, f : FitnessFunction, termination_condition):
    p = FrequencyVector(n)                         # p.vector initialized to 0.5
    t = 0
    sampler = IndividualFactory(n)
    sampler.frequency_vector = p
    x = sampler.sample_individual()

    while not termination_condition:
        y = x.mutate_individual(x)
        if f.evaluate(y) > f.evaluate(x):
            x = y

        t += 1