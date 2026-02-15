from FrequencyVector import FrequencyVector
from OriginalHistory import OriginalHistory
from Individual import *
from sigfunction import sig
from HistoryNode import HistoryNode

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


def cGA(n: int,k: int, f, termination_condition):
    p = FrequencyVector(n)                    # p.vector initialized to 0.5
    histories = [OriginalHistory(n) for _ in range(n)]  # one history per position
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