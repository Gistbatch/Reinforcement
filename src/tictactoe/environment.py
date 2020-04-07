import numpy as np

class Environment:
    def __init__(self):
        self.state = np.zeros((3, 3))
        self.is_over = False
        self.winner = 0

    def print_board(self, turn):
        states = np.reshape(self.state, 9)
        output = ''
        for index, state in enumerate(states):
            if index == 3 or index == 6:
                output += '\n_ _ _\n'
            if index % 3 != 2: 
                output += self.add_sign(state) + '|'
            else:
                output += self.add_sign(state)
        output += '\n' + str(turn) +'___________\n'
        print(output)

    def add_sign(self, value):
        if value == 1:
            return 'X'
        if value == -1:
            return 'O'
        return ' '

    def get_state_as_int(self):
        k = 0
        h = 0
        for i in range(3):
            for j in range(3):
                if self.state[i,j] == 0:
                    v = 0
                elif self.state[i,j] == 1:
                    v = 1
                elif self.state[i,j] == -1:
                    v = 2
                h += (3**k) * v
                k += 1
        return h

    def is_empty(self, i, j):
        return self.state[i][j] == 0
    
    def is_drawn(self):
        return self.is_over and not self.winner

    def reward(self, agent):
        if not self.is_over:
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

    def possibilities(self, i=0, j=0):
        results = []
        for v in (0, 1, -1):
            self.state[i,j] = v # if empty board it should already be 0
            if j == 2:
            # j goes back to 0, increase i, unless i = 2, then we are done
                if i == 2:
                    # the board is full, collect results and return
                    state = self.get_state_as_int()
                    ended = self.game_over()
                    winner = self.winner
                    results.append((state, winner, ended))
                else:
                    results += self.possibilities(i + 1, 0)
            else:
            # increment j, i stays the same
                results += self.possibilities(i, j + 1)

        return results
