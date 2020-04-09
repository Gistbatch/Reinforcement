class Gridworld:
    def __init__(self, rows, cols, start):
        self.rows = rows
        self.cols = cols
        self.i_pos = start[0]
        self.j_pos = start[1]

    def set_reward_and_actions(self, rewards, actions):
        self.rewards = rewards
        self.actions = actions

    def set_state(self, pos):
        self.i_pos = pos[0]
        self.j_pos = pos[1]

    def get_state(self):
        return (self.i_pos, self.j_pos)

    def all_states(self):
        return set(self.actions.keys()) | set(self.rewards.keys())

    def is_terminal(self, state):
        return state not in self.actions

    def game_over(self):
        return self.is_terminal((self.i_pos, self.j_pos))

    def do_action(self, action):
        if self.get_state() in self.actions.keys() and action in self.actions[
                self.get_state()]:
            if action == 'U':
                self.i_pos -= 1
            elif action == 'D':
                self.i_pos += 1
            elif action == 'L':
                self.j_pos -= 1
            elif action == 'R':
                self.j_pos += 1
            else:
                pass
        assert (self.get_state() in self.all_states())
        return self.rewards.get(self.get_state(), 0)

    def undo_action(self, action):
        if action == 'U':
            self.i_pos += 1
        elif action == 'D':
            self.i_pos -= 1
        elif action == 'L':
            self.j_pos += 1
        elif action == 'R':
            self.j_pos -= 1
        else:
            pass
        assert (self.get_state() in self.all_states())

    @staticmethod
    def default_grid():
        grid = Gridworld(3, 4, (2, 0))
        rewards = {(0, 3): 1, (1, 3): -1}
        actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('R', 'L'),
            (0, 2): ('D', 'R', 'L'),
            (1, 0): ('D', 'U'),
            (1, 2): ('D', 'U', 'R'),
            (2, 0): ('U', 'R'),
            (2, 1): ('R', 'L'),
            (2, 2): ('U', 'R', 'L'),
            (2, 3): ('U', 'L')
        }
        grid.set_reward_and_actions(rewards, actions)
        return grid

    @staticmethod
    def negative_grid(step_cost=-0.1):
        grid = Gridworld.default_grid()
        grid.rewards.update({
            (0, 0): step_cost,
            (0, 1): step_cost,
            (0, 2): step_cost,
            (1, 0): step_cost,
            (1, 2): step_cost,
            (2, 0): step_cost,
            (2, 1): step_cost,
            (2, 2): step_cost,
            (2, 3): step_cost
        })
        return grid