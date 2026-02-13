from Individual import Individual
from FrequencyVector import FrequencyVector
from OriginalHistory import OriginalHistory

h = OriginalHistory(10)
for i in range(33):
    h.add(1)

curr = h.head
while curr is not None:
    print(f"m: {curr.m}, ones: {curr.ones}, zeros: {curr.zeros}")
    curr = curr.next
