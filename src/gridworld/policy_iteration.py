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


def policy_evaluation(grid, discount, policy=None, values=None):
    """
    If policy is none use uniform distribution
    """
    states = grid.all_states()
    if not values:
        values = {}
        for state in states:
            values[state] = 0
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            if state in grid.actions:
                new_value = 0          
                if policy:
                    actions = policy[state]
                    policy_probability = 1  # possibly get distribution
                else:
                    actions = grid.actions[state]
                    policy_probability = 1 / len(actions)
                for action in actions:
                    grid.set_state(state)
                    reward = grid.do_action(action)
                    value = discount * values[grid.get_state()]
                    new_value += policy_probability * (reward + value)
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
                for action in POSSIBLE_ACTIONS:
                    grid.set_state(state)
                    reward = grid.do_action(action)
                    value = reward + discount * values[grid.get_state()]
                    if value > best_value:
                        best_value = value
                        new_action = action

                policy[state] = new_action
                if old_action != new_action:
                    policy_converged = False

    return values, policy

def simple_example():
    grid = Gridworld.default_grid()
    values_uniform = policy_evaluation(grid, 1)
    print('values for uniformly random actions:')
    print_values(values_uniform, grid)
    print('\n\n')

    fixed_policy = {
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
    print_policy(fixed_policy, grid)
    fixed_values = policy_evaluation(grid, 0.9, fixed_policy)
    print('Values for fixed policy:')
    print_values(fixed_values, grid)


def iteration_example():
    grid = Gridworld.negative_grid()
    print('grid:')
    print_values(grid.rewards, grid)
    values, policy = policy_iteration(grid, 0.9)
    print('values:')
    print_values(values, grid)
    print('policy:')
    print_policy(policy, grid)


if __name__ == '__main__':
   iteration_example()