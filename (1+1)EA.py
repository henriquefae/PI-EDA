from FrequencyVector import FrequencyVector
from Individual import Individual


def one_plus_one_EA(n: int, f, termination_condition):
    p = FrequencyVector(n)                         # p.vector initialized to 0.5
    t = 0
    sampler = Individual(n)
    sampler.frequency_vector = p
    x = sampler.sample_individual()

    while not termination_condition:
        y = sampler.mutate_individual(x)
        if f(y) > f(x):
            x = y

        t += 1