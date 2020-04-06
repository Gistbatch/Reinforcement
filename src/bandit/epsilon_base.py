import numpy as np

from bandit import Bandit

if __name__ == "__main__":
    number_bandits = 10
    bandits = [Bandit(i) for i in range(number_bandits)]
    epsilon = 0.1
    current_best = bandits[np.random.randint(0, number_bandits)]
    print(f'Starting with bandit {current_best.true_mean}.')
    for _ in range(10000):
        explore_exploit = np.random.randn()
        bandit = current_best
        # explore
        if explore_exploit < epsilon:
            selection = np.random.randint(0, number_bandits)
            #print(f'Machine {selection} selected.')
            bandit = bandits[selection]
        # exploit
        bandit.update(bandit.pull())
        #update
        if current_best.current_mean < bandit.current_mean:
            print(f'Updated to bandit {bandit.true_mean}')
            current_best = bandit

    print(f'Chose bandit {current_best.true_mean}')