class Human:
    def __init__(self, sign):
        self.sign = 1 if sign == 'X' else -1

    def take_action(self, env):
        while True:
            # break if we make a legal move
            move = input("Enter coordinates i,j for your next move (i,j=0..2): ")
            i, j = move.split(',')
            i = int(i)
            j = int(j)
            if not (0 <= i <= 2 and 0 <= j <= 2):
                continue
            if env.is_empty(i, j):
                env.state[i,j] = self.sign
                break

    def update_value(self, e):
        pass

    def update_history(self, s):
        pass