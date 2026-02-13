class FrequencyVector:
    def __init__(self, n):
        self.n = n
        self.vector = [0.5] * n  # Initialize frequency vector with 0.5 for each position

    def update_vector(self, p):
        """Update the frequency vector that satisfies the border condition"""
        self.vector = [max(1/self.n, min(1- 1/self.n, p[i])) for i in range(self.n)]
        return self.vector