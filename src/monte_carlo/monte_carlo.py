import numpy as np

from gridworld import Gridworld


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


def play_episode(grid, policy, discount=0.9):  #
    #random init game
    states = list(grid.actions.keys())
    start = np.random.choice(len(states))
    grid.set_state(states[start])

    current_state = grid.get_state()
    states_rewards = [(current_state, 0)]
    while not grid.game_over():
        action = policy[current_state]
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


def first_visit_monte_carlo(grid, iterations, policy):
    values = {}
    all_returns = {}
    states = grid.all_states()
    for state in states:
        if grid.is_terminal(state):
            values[state] = 0
        else:
            all_returns[state] = []

    for index in range(iterations):
        states_returns = play_episode(grid, policy)
        seen_states = []
        for state, ret in states_returns:
            if state not in seen_states:  #state not seen
                all_returns[state].append(ret)
                values[state] = np.mean(all_returns[state])
                seen_states.append(state)
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
    values = first_visit_monte_carlo(grid, 100, policy)
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    simple_example()