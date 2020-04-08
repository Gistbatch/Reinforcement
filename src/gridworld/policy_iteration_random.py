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


def policy_evaluation(grid, discount, policy, values):
    states = grid.all_states()
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            if state in policy:
                new_value = 0
                for action in POSSIBLE_ACTIONS:
                    if action == policy[state]:
                        action_probability = 0.5
                    else:
                        action_probability = 0.5 / 3
                    grid.set_state(state)
                    reward = grid.do_action(action)
                    value = discount * values[grid.get_state()]
                    new_value += action_probability * (reward + value)
                values[state] = new_value
                delta = max(delta, abs(new_value - old_value))
        if delta < THRESHOLD:
            break
    return values


def policy_iteration(grid, discount):
    #init
    states = grid.all_states()
    policy = {}
    values = {}
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
    policy_converged = False
    while not policy_converged:
        values = policy_evaluation(grid, discount, policy, values)
        policy_converged = True
        for state in states:
            if state in policy:
                old_action = policy[state]
                new_action = None
                best_value = float('-inf')
                for chosen_action in POSSIBLE_ACTIONS:
                    value = 0
                    for action in POSSIBLE_ACTIONS:
                        if chosen_action == action:
                            action_prob = 0.5
                        else:
                            action_prob = 0.5 / 3

                        grid.set_state(state)
                        reward = grid.do_action(action)
                        value += action_prob * (
                            reward + discount * values[grid.get_state()])
                    if value > best_value:
                        best_value = value
                        new_action = chosen_action

                policy[state] = new_action
                if old_action != new_action:
                    policy_converged = False

    return values, policy


def windy_example():
    grid = Gridworld.negative_grid(step_cost=-1.0)
    print('grid:')
    print_values(grid.rewards, grid)
    values, policy = policy_iteration(grid, 0.9)
    print('values:')
    print_values(values, grid)
    print('policy:')
    print_policy(policy, grid)


if __name__ == '__main__':
    windy_example()
