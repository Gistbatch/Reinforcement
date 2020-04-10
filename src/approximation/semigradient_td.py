import numpy as np
import matplotlib.pyplot as plt

from gridworld import Gridworld

POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def print_values(values, grid):
    for i in range(grid.rows):
        print('-----------------')
        for j in range(grid.cols):
            value = values.get((i, j), 0)
            if value >= 0:
                print(f' {value:.2f}|', end='')
            else:
                print(f'{value:.2f}|', end='')
        print('')


def print_policy(policy, grid):
    for i in range(grid.rows):
        print('-----------------')
        for j in range(grid.cols):
            action = policy.get((i, j), ' ')
            print(f' {action} |', end='')
        print('')


def random_action(action, epsilon):
    explore = np.random.rand()
    if explore < epsilon:
        return np.random.choice(POSSIBLE_ACTIONS)
    return action


class Model:
    def __init__(self):
        self.theta = np.array(np.random.randn(4)) / 2

    def features(self, state):
        return np.array(
            [state[0] - 1, state[1] - 1.5, state[0] * state[1] - 3, 1])

    def predict(self, state):
        return self.theta.dot(self.features(state))
    
    def gradient(self, state):
        return self.features(state)


def play_game(grid, policy, epsilon):
    grid.set_state((2, 0))
    current_state = grid.get_state()
    states_rewards = [(current_state, 0)]
    while not grid.game_over():
        action = random_action(policy[current_state], epsilon)
        reward = grid.do_action(action)
        current_state = grid.get_state()
        states_rewards.append((current_state, reward))
    return states_rewards

def semi_td_zero(grid,
                 policy,
                 iterations=40000,
                 epsilon=0.1,
                 discount=0.9,
                 learning_rate_0=0.01):
    model = Model()
    decay = 1.0
    deltas = []
    for index in range(iterations):
        if index % 100 == 0:
            decay += 0.01
        learning_rate = learning_rate_0 / decay
        delta = 0
        states_rewards = play_game(grid, policy, epsilon)
        for step in range(len(states_rewards) - 1):
            state_t, _ = states_rewards[step]
            state_t1, reward_t1 = states_rewards[step + 1]
            old_theta = model.theta.copy()
            if grid.is_terminal(state_t1):
                target = reward_t1
            else:
                target = reward_t1 + discount * model.predict(state_t1)
            model.theta += learning_rate * (target - model.predict(state_t)) * model.gradient(state_t)
            delta = max(delta, np.abs(old_theta - model.theta).sum())
        deltas.append(delta)
    values = {}
    states = grid.all_states()
    for state in states:
        if grid.is_terminal(state):
            values[state] = 0
        else:
            values[state] = model.predict(state)
    return values, deltas


def simple_example():
    grid = Gridworld.default_grid()
    print('Rewards:')
    print_values(grid.rewards, grid)
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
    }
    values, deltas = semi_td_zero(grid, policy)
    plt.plot(deltas)
    plt.show()
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    simple_example()