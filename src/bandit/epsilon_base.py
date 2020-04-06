import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit


def run_experiment(number_bandits, iterations, epsilon=0):
    optimistic = 10 if epsilon else 0
    bandits = [Bandit(i + 1, optimistic) for i in range(number_bandits)]
    current_best = bandits[np.random.randint(0, number_bandits)]
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


if __name__ == "__main__":
    epsilon_1 = run_experiment(3, 100000, 0.1)
    epsilon_05 = run_experiment(3, 100000, 0.05)
    epsilon_01 = run_experiment(3, 100000, 0.01)

    optimistic = run_experiment(3, 100000)

    # log scale plot
    plt.plot(epsilon_1, label='eps = 0.1')
    plt.plot(epsilon_05, label='eps = 0.05')
    plt.plot(epsilon_01, label='eps = 0.01')
    plt.plot(optimistic, label='optimistic')
    plt.legend()
    plt.xscale('log')
    plt.show()