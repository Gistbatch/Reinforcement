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


def max_dict(d):
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


def random_action(action, epsilon):
    explore = np.random.rand()
    if explore < epsilon:
        return np.random.choice(POSSIBLE_ACTIONS)
    return action


def play_episode(grid, policy, discount=0.9, epsilon=0.1):  #
    #reset grid
    grid.set_state((2, 0))
    current_state = grid.get_state()
    action = random_action(policy[current_state], epsilon)
    seen_states = set()
    states_actions_rewards = [(current_state, action, 0)]
    while True:
        reward = grid.do_action(action)
        current_state = grid.get_state()
        if grid.game_over():
            states_actions_rewards.append((current_state, None, reward))
            break
        else:
            action = random_action(policy[current_state], epsilon)
            states_actions_rewards.append((current_state, action, reward))

    ret = 0
    states_returns = []
    first = True
    for state, action, reward in reversed(states_actions_rewards):
        if first:
            first = False
        else:
            states_returns.append((state, action, ret))
        ret = reward + discount * ret
    states_returns.reverse()
    return states_returns


def policy_iteration(grid, iterations, policy):
    Q = {}
    values = {}
    all_returns = {}
    states = grid.all_states()
    for state in states:
        if grid.is_terminal(state):
            pass
        else:
            Q[state] = {}
            for action in POSSIBLE_ACTIONS:
                Q[state][action] = 0
                all_returns[(state, action)] = []

    deltas = []
    for index in range(iterations):
        if index % 200 == 0:
            print(index)
        delta = 0
        states_actions_returns = play_episode(grid, policy)
        seen_state_actions = set()
        for state, action, ret in states_actions_returns:
            state_action = (state, action)
            if state_action not in seen_state_actions:  #state not seen
                old_q = Q[state][action]
                all_returns[state_action].append(ret)
                new_q = np.mean(all_returns[state_action])
                Q[state][action] = new_q
                delta = max(delta, abs(old_q - new_q))
                seen_state_actions.add(state_action)
        deltas.append(delta)
        for state in policy.keys():
            policy[state] = max_dict(Q[state])[0]

    for state, _ in Q.items():
        values[state] = max_dict(Q[state])[1]
    return values, deltas, policy


def epsilon_soft_example():
    grid = Gridworld.negative_grid(-0.1)
    print('Rewards:')
    print_values(grid.rewards, grid)
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(POSSIBLE_ACTIONS)

    values, deltas, policy = policy_iteration(grid, 5000, policy)
    plt.plot(deltas)
    plt.show()
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    epsilon_soft_example()