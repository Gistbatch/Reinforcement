import numpy as np

class Environment:
    def __init__(self):
        self.state = np.zeros((3, 3))
        self.is_over = False
        self.winner = 0

    def print_board(self):
        states = np.reshape(self.state, 9)
        output = ''
        for index, state in enumerate(states):
            if index == 3 or index == 6:
                output += '\n_ _ _\n'
            if index % 3 != 2: 
                output += self.add_sign(state) + '|'
            else:
                output += self.add_sign(state)
        print(output)

    def add_sign(self, value):
        if value == 1:
            return 'X'
        if value == -1:
            return 'O'
        return ' '

    def get_state(self):
        return self.state

    def is_empty(self, i, j):
        return self.state[i][j] == 0
    
    def is_drawn(self):
        return self.is_over and not self.winner

    def reward(self, agent):
        if not self.is_Over:
            return 0
        return 1 if self.winner == agent.sign else 0

    def game_over(self): 
        #rows
        if 3 in np.sum(self.state, axis=1):
            self.winner = 1
            self.is_over = True
            return True
        if -3 in np.sum(self.state, axis=1):
            self.is_over = True
            self.winner = -1
            return True
        #cols
        if 3 in np.sum(self.state, axis=0):
            self.winner = 1
            self.is_over = True
            return True
        if -3 in np.sum(self.state, axis=0):
            self.winner = -1
            self.is_over = True
            return True
        #diag
        if 3 ==  np.sum(np.diag(self.state)):
            self.winner = 1
            self.is_over = True
            return True
        if -3 ==  np.sum(np.diag(self.state)):
            self.winner = -1
            self.is_over = True
            return True
        #reverse diag
        if 3 == np.sum(np.diag(np.fliplr(self.state))):
            self.winner = 1
            self.is_over = True
            return True
        if -3 == np.sum(np.diag(np.fliplr(self.state))):
            self.winner = -1
            self.is_over = True
            return True
        
        if 0 not in self.state:
            self.is_over = True
            self.winner = 0
            return True
        self.winner = 0
        return False

if __name__ == "__main__":
    env = Environment()
    env.print_board()
    print(env.is_over())