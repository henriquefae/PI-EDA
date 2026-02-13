import numpy as np

def sig(p: float,h,e:float, n: int) -> str:
    bound = p*h.m+e*max(np.ln(n), np.sqrt(np.ln(n)*p*h.m))
    
    if (p == 1 / n or p == 1 / 2) and h.ones >= bound:
        return "up"
    elif (p == 1 / n or p == 1 / 2) and h.zeros >= bound:
        return "down"
    else:
        return "stay"