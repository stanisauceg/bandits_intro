import math
import random
import warnings


class Solver:
    @staticmethod
    def ind_max(x):
        m = max(x)
        return x.index(m)

    def __init__(self, counts, values):
        self.counts = counts
        self.values = values

    def initialize(self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]

    def select_arm(self):
        raise NotImplementedError

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value


class EpsilonGreedy(Solver):
    def __init__(self, epsilon, counts, values):
        super().__init__(counts, values)
        self.epsilon = epsilon

    def select_arm(self):
        if random.random() > self.epsilon:
            return self.ind_max(self.values)
        else:
            return random.randrange(len(self.values))


class Softmax(Solver):

    @staticmethod
    def categorical_draw(probs):
        z = random.random()
        cum_prob = 0.0
        for i in range(len(probs)):
            prob = probs[i]
            cum_prob += prob
            if cum_prob > z:
                return i
        return len(probs) - 1

    def __init__(self, temperature, counts, values):
        super().__init__(counts, values)
        self.temperature = temperature

    def select_arm(self):
        z = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) / z for v in self.values]
        return self.categorical_draw(probs)


class AnnealingSoftmax(Softmax):
    def select_arm(self):
        t = sum(self.counts) + 1
        temperature = 1 / math.log(t + 0.0000001)

        z = sum([math.exp(v / temperature) for v in self.values])
        probs = [math.exp(v / temperature) / z for v in self.values]
        return self.categorical_draw(probs)


class UCB1(Solver):
    def __init__(self):
        warnings.warn("UCB1 assumes rewards range from 0 to 1")
        super().__init__()

    def select_arm(self):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                return arm
            ucb_values = [0.0 for arm in range(n_arms)]
            total_counts = sum(self.counts)
            for arm in range(n_arms):
                bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
                ucb_values[arm] = self.values[arm] + bonus
            return self.ind_max(ucb_values)


