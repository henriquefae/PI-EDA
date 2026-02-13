import numpy as np
from HistoryNode import HistoryNode

def sig(p: float,h: HistoryNode,e:float, n: int) -> str:
    m = h.m
    bound = p*m+e*max(np.log(n), np.sqrt(np.log(n)*p*m))

    if (p == 1 / n or p == 1 / 2) and h.ones >= bound:
        return "up"
    elif (p == 1 / n or p == 1 / 2) and h.zeros >= bound:
        return "down"
    else:
        return "stay"