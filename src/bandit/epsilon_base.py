import math

import matplotlib.pyplot as plt
import numpy as np

from bandit import Bandit
from bayesian import Bayesian


def ucb(mean, n, n_j):
    if n_j:
        return mean + math.sqrt(2 * math.log(n) / n_j)
    return float('inf')


def run_greedy(number_bandits, epsilon, iterations):
    bandits = [Bandit(i + 1, 0) for i in range(number_bandits)]
    current_best = bandits[0]
    data = np.empty(iterations)
    print(f'Starting with bandit {current_best.true_mean}.')
    for i in range(iterations):
        explore_exploit = np.random.rand()
        bandit = current_best
        # explore
        if explore_exploit < epsilon:
            selection = np.random.randint(0, number_bandits)
            #print(f'Machine {selection} selected.')
            bandit = bandits[selection]
        # exploit
        value = bandit.pull()
        bandit.update(value)
        data[i] = value
        #update
        if current_best.current_mean < bandit.current_mean:
            print(f'Updated to bandit {bandit.true_mean}')
            current_best = bandit

    print(f'Chose bandit {current_best.true_mean}')
    cumulative_average = np.cumsum(data) / (np.arange(iterations) + 1)
    plt.plot(cumulative_average)
    for i in range(number_bandits):
        plt.plot(np.ones(iterations) * (i + 1))
    plt.xscale('log')
    plt.show()
    return cumulative_average


def run_optimistic(number_bandits, iterations):
    bandits = [Bandit(i + 1, 10) for i in range(number_bandits)]
    data = np.empty(iterations)
    for i in range(iterations):
        bandit = bandits[np.argmax([bandit.current_mean
                                    for bandit in bandits])]
        # exploit
        value = bandit.pull()
        bandit.update(value)
        data[i] = value

    cumulative_average = np.cumsum(data) / (np.arange(iterations) + 1)
    plt.plot(cumulative_average)
    for i in range(number_bandits):
        plt.plot(np.ones(iterations) * (i + 1))
    plt.xscale('log')
    plt.show()
    return cumulative_average


def run_ucb(number_bandits, iterations):
    bandits = [Bandit(i + 1, 10) for i in range(number_bandits)]
    data = np.empty(iterations)
    for i in range(iterations):
        bandit = bandits[np.argmax([
            ucb(bandit.current_mean, i, bandit.times_pulled)
            for bandit in bandits
        ])]
        # exploit
        value = bandit.pull()
        bandit.update(value)
        data[i] = value

    cumulative_average = np.cumsum(data) / (np.arange(iterations) + 1)
    plt.plot(cumulative_average)
    for i in range(number_bandits):
        plt.plot(np.ones(iterations) * (i + 1))
    plt.xscale('log')
    plt.show()
    return cumulative_average


def run_decay(number_bandits, iterations):
    bandits = [Bandit(i + 1, 0) for i in range(number_bandits)]
    data = np.empty(iterations)
    for i in range(iterations):
        bandit = bandits[np.argmax([bandit.current_mean for bandit in bandits])]
        explore_exploit = np.random.rand()
        # explore
        if explore_exploit < 1/(i+1):
            bandit = bandits[np.random.choice(number_bandits)]

        # exploit
        value = bandit.pull()
        bandit.update(value)
        data[i] = value

    cumulative_average = np.cumsum(data) / (np.arange(iterations) + 1)
    plt.plot(cumulative_average)
    for i in range(number_bandits):
        plt.plot(np.ones(iterations) * (i + 1))
    plt.xscale('log')
    plt.show()
    return cumulative_average


def run_bayesian(number_bandits, iterations):
    bandits = [Bayesian(i + 1) for i in range(number_bandits)]
    data = np.empty(iterations)
    for i in range(iterations):
        bandit = bandits[np.argmax([bandit.sample() for bandit in bandits])]
        value = bandit.pull()
        bandit.update(value)
        data[i] = value

    cumulative_average = np.cumsum(data) / (np.arange(iterations) + 1)
    plt.plot(cumulative_average)
    for i in range(number_bandits):
        plt.plot(np.ones(iterations) * (i + 1))
    plt.xscale('log')
    plt.show()
    return cumulative_average


if __name__ == "__main__":
    epsilon_1 = run_greedy(3, 0.1, 100000)
    epsilon_05 = run_greedy(3, 0.05, 100000)
    epsilon_01 = run_greedy(3, 0.01, 100000)

    optimistic = run_optimistic(3, 100000)
    ucb = run_ucb(3, 100000)
    decay = run_decay(3, 100000)
    bayesian = run_bayesian(3, 100000)

    # log scale plot
    plt.plot(epsilon_1, label='eps = 0.1')
    plt.plot(epsilon_05, label='eps = 0.05')
    plt.plot(epsilon_01, label='eps = 0.01')
    plt.plot(optimistic, label='optimistic')
    plt.plot(ucb, label='ucb')
    plt.plot(decay, label='decay')
    plt.plot(bayesian, label='bayesian')
    plt.legend()
    plt.xscale('log')
    plt.show()
