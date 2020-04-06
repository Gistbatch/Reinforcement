import numpy as np

from bandit import Bandit

if __name__ == "__main__":
    number_bandits = 10
    bandits = [Bandit(i) for i in range(number_bandits)]
    epsilon = 0.1
    current_best = bandits[np.random.randint(0, number_bandits)]
    for _ in range(1000):
        explore_exploit = np.random.random_sample()
        bandit = current_best
        # explore
        if explore_exploit < epsilon:
            selection = np.random.randint(0, number_bandits)
            print(f'Machine {selection} selected.')
            bandit = bandits[selection]
        # exploit
        bandit.update_avg(bandit.pull())
        #update
        if current_best.current_avg < bandit.current_avg:
            print(f'Updated to bandit {bandit.true_avg}')
            current_best = bandit

    print(f'Chose bandit {current_best.true_avg}')