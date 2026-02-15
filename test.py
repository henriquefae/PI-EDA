from OriginalHistory import OriginalHistory

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
