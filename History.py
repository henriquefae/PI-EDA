class History:
    def __init__(self):
        self.m = 0
        self.ones = 0
        self.zeros = 0
    
    def add(self, info):
        if info == 1:
            self.ones += 1
        elif info == 0:
            self.zeros += 1
        else:
            raise ValueError("Info must be either 0 or 1.")
        self.m += 1

    def reset(self):
        self.m = 0
        self.ones = 0
        self.zeros = 0