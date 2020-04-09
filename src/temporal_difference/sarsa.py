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


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def sarsa(grid, iterations=10000, discount=0.9, alpha_0=0.1):
    states = grid.all_states()
    Q = {}
    counts = {}
    count_updates = {}
    for state in states:
        Q[state] = {}
        counts[state] = {}
        for action in POSSIBLE_ACTIONS:
            Q[state][action] = 0
            counts[state][action] = 1.0

    decay = 1.0
    deltas = []
    for step in range(iterations):
        delta = 0
        if step % 100 == 0:
            decay += 1e-2
            epsilon = 0.5 / decay
        if step % 1000 == 0:
            print(f'Step {step}')
        state_t = (2, 0)
        action_t = random_action(max_dict(Q[state_t])[0], epsilon)
        grid.set_state(state_t)

        while not grid.game_over():
            reward_t = grid.do_action(action_t)
            state_t1 = grid.get_state()
            action_t1 = random_action(max_dict(Q[state_t1])[0], epsilon)
            alpha = alpha_0 / counts[state_t][action_t]
            old_qsa = Q[state_t][action_t]
            Q[state_t][action_t] = old_qsa + alpha * (
                reward_t + discount * Q[state_t1][action_t1] - old_qsa)
            counts[state_t][action_t] += 0.005
            count_updates[state_t] = count_updates.get(state_t, 0) + 1
            delta = max(delta, abs(old_qsa - Q[state_t][action_t]))
            state_t = state_t1
            action_t = action_t1
        deltas.append(delta)
    values = {}
    policy = {}
    for state in states:
        if not grid.is_terminal(state):
            action, value = max_dict(Q[state])
            policy[state] = action
            values[state] = value
    total_count = np.sum(list(count_updates.values()))
    for key, val in count_updates.items():
        count_updates[key] = val / total_count
    return values, policy, deltas, count_updates


def sarsa_example():
    grid = Gridworld.negative_grid()
    print('Rewards:')
    print_values(grid.rewards, grid)
    values, policy, deltas, count_updates = sarsa(grid)
    plt.plot(deltas)
    plt.show()
    print('Updates:')
    print_values(count_updates, grid)
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)

if __name__ == "__main__":
    sarsa_example()