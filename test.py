from FitnessFunctions import OneMax
from Individual import Individual
from FrequencyVector import FrequencyVector
from OriginalHistory import OriginalHistory
from TerminationCriterion import TerminationCriterion
from cGA import cGA
from sigcGA import sigcGA
from onePlusOneEA import onePlusOneEA

n = 100          # problem size
k = 50           # hypothetical population size
budget = 100000  # max evaluations

# create fitness function
fitness = OneMax()

# create termination criterion
termination = TerminationCriterion(
    fitness=fitness,
    n=n,
    budget=budget
)

p, t = onePlusOneEA(n, fitness, termination)

print(f"\nTest results for (1+1) EA with {fitness.__class__.__name__} as fitness function:")
print("Finished after iterations:", t)
print("Final frequencies:", p.vector)
print("Total evaluations:", fitness.call_count)

p, t = cGA(n, k, fitness, termination)

print(f"\nTest results for cGA with {fitness.__class__.__name__} as fitness function:")
print("Finished after iterations:", t)
print("Final frequencies:", p.vector)
print("Total evaluations:", fitness.call_count)

p, t = sigcGA(n, 0.05, fitness, termination)

print("\nTest results for sigcGA:")
print("Finished after iterations:", t)
print("Final frequencies:", p.vector)
print("Total evaluations:", fitness.call_count)

# h = OriginalHistory(10)
# for i in range(33):
#     h.add(1)

# curr = h.head
# while curr is not None:
#     print(f"m: {curr.m}, ones: {curr.ones}, zeros: {curr.zeros}")
#     curr = curr.next
class Test:
    def test_history_consolidation():
        h = OriginalHistory(10)
        for i in range(33):
            h.add(1)

        curr = h.head
        while curr is not None:
            print(f"m: {curr.m}, ones: {curr.ones}, zeros: {curr.zeros}")
            curr = curr.next

    def test_benchmark_functions():
        pass

Test.test_history_consolidation()
