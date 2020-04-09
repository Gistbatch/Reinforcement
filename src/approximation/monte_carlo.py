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


def random_action(chosen_action):
    rng = np.random.rand()
    if rng < 0.5:
        return (chosen_action)
    remaining_actions = list(POSSIBLE_ACTIONS)
    remaining_actions.remove(chosen_action)
    return np.random.choice(remaining_actions)


def play_episode(grid, policy, discount=0.9):
    #random init game
    states = list(grid.actions.keys())
    start = np.random.choice(len(states))
    grid.set_state(states[start])

    current_state = grid.get_state()
    states_rewards = [(current_state, 0)]
    while not grid.game_over():
        chosen_action = policy[current_state]
        action = random_action(chosen_action)

        reward = grid.do_action(action)
        current_state = grid.get_state()
        states_rewards.append((current_state, reward))

    ret = 0
    states_returns = []
    first = True
    for state, reward in reversed(states_rewards):
        if first:
            first = False
        else:
            states_returns.append((state, ret))
        ret = reward + discount * ret
    states_returns.reverse()
    return states_returns


def feature(state):
    return np.array([state[0] - 1, state[1] - 1.5, state[0] * state[1] - 3, 1])


def approx_monte_carlo(grid, policy, iterations=20000, learning_rate_0=0.1):
    theta =np.array(np.random.randn(4))
    decay = 1.0
    deltas = []
    for index in range(iterations):
        if index % 100 == 0:
            decay += 0.01
        learning_rate = learning_rate_0 / decay
        delta = 0
        states_returns = play_episode(grid, policy)
        seen_states = set()
        for state, ret in states_returns:
            if state not in seen_states:  #state not seen
                old_theta = theta.copy()
                x = feature(state)
                theta += learning_rate * (ret - theta.dot(x)) * x
                delta = max(delta, np.abs(old_theta - theta).sum())
                seen_states.add(state)
        deltas.append(delta)
    values = {}
    states = grid.all_states()
    for state in states:
        if grid.is_terminal(state):
            values[state] = 0
        else:
            values[state] = theta.dot(feature(state))
    return values, deltas


def mc_prediction():
    grid = Gridworld.default_grid()
    print('Rewards:')
    print_values(grid.rewards, grid)
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }
    values, deltas = approx_monte_carlo(grid, policy)
    plt.plot(deltas)
    plt.show()
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    mc_prediction()