from FrequencyVector import FrequencyVector
from History import History
from Individual import Individual
from sigfunction import sig

def sigcGA(p: FrequencyVector, histories: list[History], e: float, f, termination_condition):
    """sig-cGA algorithm"""
    n = len(p)
    t = 0
    for i in range(n):
        histories[i] = History.reset()
    while not termination_condition(t):
        x_1, x_2 = Individual(n).sample_individual(), Individual(n).sample_individual()
        if f(x_1) > f(x_2):
            x = x_1
        else:
            x = x_2
        for i in range(n):
            histories[i].add(x[i])
            h = histories[i]
            s = sig(p[i], h, e, n)
            match s:
                case "up":
                    p[i] = 1 - 1 / n
                case "down":
                    p[i] = 1 / n
            if s != "stay":
                histories[i].reset()
                break
        t += 1