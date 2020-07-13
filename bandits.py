import random
from solvers import Solver, EpsilonGreedy, Softmax, AnnealingSoftmax


class BernoulliArm:
    """
    For use in an experiment where each round is IID with probability of success p and probability of failure 1-p
    """
    def __init__(self, p):
        """
        :param p: the probability of success on any given draw
        """
        self.p = p

    def draw(self):
        """
        this method returns a success with probability p, and failure with probability 1-p
        :return: 1.0 (success) or 0.0 (failure)
        """
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0


# def example():
#     means = [0.1, 0.1, 0.1, 0.1, 0.9]
#     # n_arms = len(means)
#     random.shuffle(means)
#     arms = list(map(lambda mu: BernoulliArm(mu), means))
#     arms[0].draw()
#     arms[1].draw()
#     arms[2].draw()
#     arms[2].draw()
#     arms[2].draw()
#     arms[3].draw()
#     arms[4].draw()


def test_algorithm(algo, arms, num_sims, horizon):
    chosen_arms = [0.0 for i in range(num_sims * horizon)]
    rewards = [0.0 for i in range(num_sims * horizon)]
    cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
    sim_nums = [0.0 for i in range(num_sims * horizon)]
    times = [0.0 for i in range(num_sims * horizon)]

    for sim in range(num_sims):
        sim += 1
        algo.initialize(len(arms))

        for t in range(horizon):
            t += 1
            index = (sim - 1) * horizon + t - 1

            sim_nums[index] = sim
            times[index] = t

            chosen_arm = algo.select_arm()
            chosen_arms[index] = chosen_arm

            reward = arms[chosen_arms[index]].draw()
            rewards[index] = reward

            if t == 1:
                cumulative_rewards[index] = reward
            else:
                cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

            algo.update(chosen_arm, reward)

            return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]


def main(fname, means, rnd_seed=1):
    random.seed(rnd_seed)
    n_arms = len(means)
    random.shuffle(means)
    arms = list(map(lambda mu: BernoulliArm(mu), means))
    print("Best arm is " + str(Solver.ind_max(means)))

    f = open(fname, "w")

    for temperature in [0.1, 0.2, 0.3, 0.4, 0.5]:
        algo = Softmax(temperature, [], [])
        algo.initialize(n_arms)
        results = test_algorithm(algo, arms, 5000, 250)
        for i in range(len(results[0])):
            f.write(str(temperature) + "\t")
            f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")

    f.close()


if __name__ == '__main__':
    main(
        fname="algorithms/softmax/standard_softmax_results.tsv",
        means=[0.1, 0.1, 0.1, 0.1, 0.9]
    )
