from gridworld import Gridworld

THRESHOLD = 1e-3


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


def policy_iteration(grid, discount, policy=None):
    """
    If policy is none use uniform distribution
    """
    values = {}
    states = grid.all_states()

    for state in states:
        values[state] = 0
    while True:
        delta = 0
        for state in states:
            old_value = values[state]
            if state in grid.actions:
                new_value = 0
                grid.set_state(state)
                if policy:
                    actions = policy[state]
                    policy_probability = 1  # possibly get distribution
                else:
                    actions = grid.actions[state]
                    policy_probability = 1 / len(actions)
                for action in actions:
                    reward = grid.do_action(action)
                    value = discount * values[grid.get_state()]
                    new_value += policy_probability * (reward + value)
                    grid.undo_action(action)
                values[state] = new_value
                delta = max(delta, abs(new_value - old_value))
        if delta < THRESHOLD:
            break
    return values


if __name__ == '__main__':
    grid = Gridworld.default_grid()
    values_uniform = policy_iteration(grid, 1)
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
    fixed_values = policy_iteration(grid, 0.9, fixed_policy)
    print('Values for fixed policy:')
    print_values(fixed_values, grid)