import numpy as np

from gridworld import Gridworld

THRESHOLD = 1e-3
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


def valule_iteration(grid, discount):
    states = grid.all_states()
    values = {}
    policy = {}
    for state in grid.actions.keys():
        policy[state] = np.random.choice(POSSIBLE_ACTIONS)
    print("initial policy:")
    print_policy(policy, grid)
    for state in states:
        if grid.is_terminal(state):
            values[state] = 0
        else:
            values[state] = np.random.rand()
    print('initial value:s')
    print_values(values, grid)
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            if state in policy:
                new_value = float('-inf')
                for action in POSSIBLE_ACTIONS:
                    grid.set_state(state)
                    reward = grid.do_action(action)
                    value = reward + discount * values[grid.get_state()]
                    if value > new_value:
                        new_value = value
                values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
        if delta < THRESHOLD:
            break

    for state in policy.keys():
        best_value = float('-inf')
        best_action = None
        for action in POSSIBLE_ACTIONS:
            grid.set_state(state)
            reward = grid.do_action(action)
            value = reward + discount * values[grid.get_state()]
            if value > best_value:
                best_value = value
                best_action = action
        policy[state] = best_action

    return policy, values


def simple_example():
    grid = Gridworld.negative_grid()
    print('rewards:')
    print_values(grid.rewards, grid)
    policy, values = valule_iteration(grid, 0.9)
    print('values:')
    print_values(values, grid)
    print('policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    simple_example()