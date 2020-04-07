from agent import Agent
from environment import Environment
from human import Human

def play_game(player1, player2, game, draw=False, turn=0):
    current_player = None
    while not game.game_over():
        if current_player == player1:
            current_player = player2
        else:
            current_player = player1
        if draw:
            game.print_board(turn)
        current_player.take_action(game)
        turn += 1
        state = game.get_state_as_int()
        player1.update_history(state)
        player2.update_history(state)
    player1.update_value(game)
    player2.update_value(game)
    if draw:
        winner = 'Player 1' if game.winner == 1 else 'Player 2'
        game.print_board(turn)
        if game.is_drawn():
            print('Draw!')
        else:
            print(f'{winner} wins!')

if __name__ == "__main__":
    player1 = Agent('X')
    player2 = Agent('O')

    game = Environment()
    player1.init_value(game)
    player2.init_value(game)

    for turn in range(10000):
        if turn % 200 == 0:
            print(turn)
        play_game(player1, player2, Environment())
    
    human = Human('O')
    while True:
        play_game(player1, human, Environment(), draw=2)
        answer = input("Play again? [Y/n]: ")
        if answer and answer.lower()[0] == 'n':
            break