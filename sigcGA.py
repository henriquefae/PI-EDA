from FrequencyVector import FrequencyVector
from OriginalHistory import OriginalHistory
from Individual import *
from TerminationCriterion import TerminationCriterion
from sigfunction import sig
from HistoryNode import HistoryNode
from FitnessFunctions import FitnessFunction

def scan_history_for_significance(p_i: float, history: OriginalHistory, e: float, n: int) -> str:
    m = 0
    ones = 0
    zeros = 0

    curr = history.head
    while curr is not None:
        m += curr.m
        ones += curr.ones
        zeros += curr.zeros

        tmp = HistoryNode()
        tmp.m = m
        tmp.ones = ones
        tmp.zeros = zeros

        s = sig(p_i, tmp, e, n)
        if s != "stay":
            return s

        curr = curr.next

    return "stay"


def sigcGA(n: int, e: float, f : FitnessFunction, termination_condition: TerminationCriterion, py_rng, np_rng)-> tuple[FrequencyVector, int]:
    p = FrequencyVector(n)                         # p.vector initialized to 0.5
    histories = [OriginalHistory(n) for _ in range(n)]  # one history per position
    t = 0

    while True:
        # sample two individuals according to p
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

        # update each position's history and potentially update p_i
        for i in range(n):
            histories[i].add(x1.genome[i])

            s = scan_history_for_significance(p.vector[i], histories[i], e, n)

            if s == "up":
                p.vector[i] = 1 - 1/n
            elif s == "down":
                p.vector[i] = 1/n

            if s != "stay":
                histories[i].reset()

        t += 1