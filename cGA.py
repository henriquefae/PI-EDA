from FrequencyVector import FrequencyVector
from Individual import *


def cGA(n: int,k: int, f, termination_condition):
    p = FrequencyVector(n)                    # p.vector initialized to 0.5
    t = 0

    while not termination_condition:
        sampler = IndividualFactory(n)
        sampler.frequency_vector = p
        x1 = sampler.sample_individual()
        x2 = sampler.sample_individual()

        # choose winner x = argmax f
        fx1 = f(x1)
        fx2 = f(x2)
        if fx2 > fx1:
            x1, x2 = x2, x1

        newP = []
        for i in range(n):
            newP.append(p[i] + (x1[i] - x2[i]) / k)
        p.update_vector(newP)

        t += 1