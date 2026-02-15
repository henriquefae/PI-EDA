from Individual import Individual


def OneMax(x: Individual) -> int:
    ''' OneMax function: counts the number of ones in the individual's vector. '''
    return sum(x.vector)

def LeadingOnes(x: Individual) -> int:
    ''' LeadingOnes function: counts the number of leading ones in the individual's vector. '''
    count = 0
    for bit in x.vector:
        if bit == 1:
            count += 1
        else:
            break
    return count

def Jump(x: Individual, k: int) -> int:
    ''' Jump function: returns the number of ones + k if the number of ones is less than n-k or equal to n, otherwise returns n - number of ones. '''
    num_ones = sum(x.vector)
    n = len(x.vector)
    if num_ones < n - k or num_ones == n:
        return num_ones + k
    else:
        return n - num_ones
