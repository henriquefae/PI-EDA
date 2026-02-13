from HistoryNode import HistoryNode
import numpy as np

class OriginalHistory:
    def __init__(self, n):
        self.m = 0
        self.n = n
        self.head = HistoryNode()
    
    def add(self, info):
        if self.head.m >= np.log(self.n):
            newHead = HistoryNode()
            newHead.next = self.head
            self.head = newHead
        self.head.add(info)
        self.m += 1
        self._consolidation()

    def _consolidation(self):
        curr = self.head
        if curr is None:
            return
        next = curr.next
        alreadySeenDoubles = False
        while curr.next is not None:
            next = curr.next
            if curr.m == next.m:
                if alreadySeenDoubles == True:
                    curr.mergeWithNext()
                    alreadySeenDoubles = False
                else:
                    alreadySeenDoubles = True
                    curr = next
            else:
                alreadySeenDoubles = False
                curr = next

    def reset(self):
        self.m = 0
        self.head = HistoryNode()