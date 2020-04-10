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


def getQs(model, state):
    Q_s = {}
    for action in POSSIBLE_ACTIONS:
        q_sa = model.predict(state, action)
        Q_s[action] = q_sa
    return Q_s


class Model:
    def __init__(self):
        self.theta = np.random.randn(25) / np.sqrt(25)

    def features(self, s, a):
        return np.array([
            s[0] - 1 if a == 'U' else 0, s[1] - 1.5 if a == 'U' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'U' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'U' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0, 1 if a == 'U' else 0,
            s[0] - 1 if a == 'D' else 0, s[1] - 1.5 if a == 'D' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'D' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'D' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0, 1 if a == 'D' else 0,
            s[0] - 1 if a == 'L' else 0, s[1] - 1.5 if a == 'L' else 0,
            (s[0] * s[1] - 3) / 3 if a == 'L' else 0,
            (s[0] * s[0] - 2) / 2 if a == 'L' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0, 1 if a == 'L' else 0,
            s[0] - 1 if a == 'R' else 0, s[1] - 1.5 if a == 'R' else 0,
            (s[0] * s[1] - 3) /
            3 if a == 'R' else 0, (s[0] * s[0] - 2) / 2 if a == 'R' else 0,
            (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0, 1 if a == 'R' else 0,
            1
        ])

    def predict(self, state, action):
        return self.theta.dot(self.features(state, action))

    def gradient(self, state, action):
        return self.features(state, action)


def gradient_q(grid, iterations=20000, discount=0.9, alpha_0=0.1):
    decay = 1.0
    deltas = []
    model = Model()
    for step in range(iterations):
        delta = 0
        if step % 100 == 0:
            decay += 1e-2
        epsilon = 0.5 / decay
        learning_rate = alpha_0 / decay
        state_t = (2, 0)
        Q_s = getQs(model, state_t)
        action_t = random_action(max_dict(Q_s)[0], epsilon)
        grid.set_state(state_t)

        while not grid.game_over():
            old_theta = model.theta.copy()
            reward_t = grid.do_action(action_t)
            state_t1 = grid.get_state()
            if grid.is_terminal(state_t1):
                model.theta += learning_rate * (reward_t - model.predict(
                    state_t, action_t)) * model.gradient(state_t, action_t)
            else:
                action_t1, max_val = max_dict(getQs(model, state_t1))#we take te best value to update
                action_t1 = random_action(action_t1, epsilon)
                model.theta += learning_rate * (
                    (reward_t + discount *  max_val)
                    - model.predict(state_t, action_t)) * model.gradient(
                        state_t, action_t)
                state_t = state_t1
                action_t = action_t1

            delta = max(delta, np.abs(old_theta - model.theta).sum())

        deltas.append(delta)

    values = {}
    policy = {}    
    states = grid.all_states()
    for state in states:
        if not grid.is_terminal(state):
            action, value = max_dict(getQs(model, state))
            policy[state] = action
            values[state] = value

    return values, policy, deltas


def q_example():
    grid = Gridworld.negative_grid()
    print('Rewards:')
    print_values(grid.rewards, grid)
    values, policy, deltas = gradient_q(grid)
    plt.plot(deltas)
    plt.show()
    print('Values:')
    print_values(values, grid)
    print('Policy:')
    print_policy(policy, grid)


if __name__ == "__main__":
    q_example()