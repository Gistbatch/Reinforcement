class Human:
    def __init__(self, sign):
        self.sign = 1 if sign is 'X' else -1

    def take_action(self, env):
        while True:
            # break if we make a legal move
            move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if env.is_empty(i, j):
                env.state[i,j] = self.sign
            break

    def update(self, e):
        pass

    def update_state_history(self, s):
        pass