from Individual import Individual

def LeadingOnes(individual: Individual) -> float:
    count = 0
    for bit in individual.genome:
        if bit == 1:
            count += 1
        else:
            break
    return count