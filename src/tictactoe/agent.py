import numpy as np


class Agent:
    def __init__(self, sign, alpha=0.5, epsilon=0.1):
        self.alpha = alpha
        self.epsilon = epsilon
        self.sign = 1 if sign is 'X' else -1
        self.history = []
        self.values = np.Zeros(3**9)

    def update_history(self, state):
        self.history.append(state)

    def reset_history(self):
        self.history = []

    def init_value(self, env):
        for state, winner, ended in env.possibilites():
            if ended:
                if winner == self.sign:
                    val = 1
                else:
                    val = 0
            else:
                val = 0.5
            self.values[state] = val

    def take_action(self, env):
        explore = np.random.rand()
        if explore < self.epsilon:
            possible_moves = []
            for i in range(3):
                for j in range(3):
                    if env.is_empty(i, j):
                        possible_moves.append((i, j))
            next_action = possible_moves[np.random.choice(len(possible_moves))]
        else:
            next_action = None
            current_best = -1
            for i in range(3):
                for j in range(3):
                    if env.is_empty(i, j):
                        env.state[i, j] = self.sign
                        state = env.get_state_as_int()
                        env.state[i, j] = 0
                        if self.values[state] > current_best:
                            current_best = self.values[state]
                            next_action = (i, j)
        env.state[next_action[0], next_action[1]] = self.sign

    def update_value(self, env):
        reward = env.reward(self.sign)
        target = reward  # reward of following state t+1 init with terminal state
        for state in self.history.reverse():
            val = self.values[state] + self.alpha * (target -
                                                     self.values[state])
            self.values[state] = val
            target = val
        self.reset_history()