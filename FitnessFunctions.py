from Individual import Individual

class FitnessFunction:

    def __init__(self):
        self.call_count = 0

    def evaluate(self, x: Individual) -> float:
        pass


class OneMax(FitnessFunction):

    ''' OneMax function: counts the number of ones in the individual's vector. '''
    def evaluate(self, x: Individual) -> float:
        self.call_count += 1
        return sum(x.genome)

class LeadingOnes(FitnessFunction):

    ''' LeadingOnes function: counts the number of leading ones in the individual's vector. '''
    def evaluate(self,x: Individual) -> float:
        self.call_count += 1
        count = 0
        for bit in x.genome:
            if bit == 1:
                count += 1
            else:
                break
        return count

class Jump(FitnessFunction):

    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    ''' Jump function: returns the number of ones + k if the number of ones is less than n-k or equal to n, otherwise returns n - number of ones. '''
    def evaluate(self, x: Individual) -> float:
        self.call_count += 1
        num_ones = sum(x.genome)
        n = len(x.genome)
        if num_ones < n - self.k or num_ones == n:
            return num_ones + self.k
        else:
            return n - num_ones
