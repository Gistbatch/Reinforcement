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


def td_zero(grid,
            policy,
            iterations=1000,
            discount=0.9,
            alpha=0.1,
            epsilon=0.1):
    values = {}
    states = grid.all_states()
    for state in states:
        if grid.is_terminal(state):
            values[state] = 0
        else:
            values[state] = 0#np.random.rand()

    for index in range(iterations):
        states_rewards = play_game(grid, policy, epsilon)
        for step in range(len(states_rewards) - 1):
            state_t, _ = states_rewards[step]
            state_t1, reward_t1 = states_rewards[step + 1]
            old_value = values[state_t]
            value = reward_t1 + discount * values[state_t1]
            values[state_t] = old_value + alpha * (value - old_value)
    return values


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
    values = td_zero(grid, policy)
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    simple_example()