import random
import time
import torch
import chess
import chess.svg

from state import State
from train_neural_network import ComplexNeuralNetwork
from train_neural_network import MediumNeuralNetwork
from train_neural_network import SimpleNeuralNetwork

class Engine(object):
    def __init__(self, model_path, model_type):
        if model_type == 'complex':
            self.model = ComplexNeuralNetwork()
        elif model_type == 'medium':
            self.model = MediumNeuralNetwork()
        else:
            self.model = SimpleNeuralNetwork()

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model_type = model_type

    def __call__(self, state):
        if self.model_type == 'complex':
            board_tensor = state.serialize_complex()[None]
        elif self.model_type == 'medium':
            board_tensor = state.serialize_medium()[None]
        else:
            board_tensor = state.serialize_simple()[None]

        output = self.model(torch.tensor(board_tensor).float())
        return float(output.data[0][0])

def explore_leaves(state, engine):
    moves = []
    for move in state.legal_moves():
        if state.make_move(move):
            moves.append((engine(state), move))
            state.board.pop()
    return moves

def computer_move_server(state, engine):
    moves = sorted(explore_leaves(state, engine), key=lambda x: x[0], reverse=state.board.turn)
    if len(moves) == 0:
      return

    print("Top 3 moves:")
    for i,m in enumerate(moves[0:3]):
        print("      ",m)
    state.make_move(moves[0][1])

    return moves[0:3]


def computer_move(state, engine):
    moves = sorted(explore_leaves(state, engine), key=lambda x: x[0], reverse=state.board.turn)
    if len(moves) == 0:
        return

    if state.board.is_repetition(2):
        if len(moves) >= 10:
            random_move = random.choice(moves[0:10])[1]
        elif len(moves) >= 5:
            random_move = random.choice(moves[0:5])[1]
        elif len(moves) >= 2:
            random_move = random.choice(moves[0:2])[1]
        else:
            random_move = moves[0][1]
        state.make_move(random_move)
    else:
        if len(moves) >= 3:
            best_move = random.choice(moves[0:3])[1]
        elif len(moves) >= 2:
            best_move = random.choice(moves[0:2])[1]
        else:
            best_move = moves[0][1]
        state.make_move(best_move)

    return moves[0:10]

def play_game(engine_white, engine_black):
    state = State()
    move_counter = 0
    while not state.is_game_over():
        if state.board.turn:
            computer_move(state, engine_white)
        else:
            computer_move(state, engine_black)
        move_counter += 1
    final_board = state.board
    return (str(state.board.outcome().winner) + " " + str(state.board.outcome().result()) + '\n' + str(final_board) + '\n'
            + 'Number of moves made: ' + str(move_counter) + '\n' + str(final_board.outcome().termination), move_counter)

if __name__ == "__main__":
    # Load models_complex
  #  simple_engine = Engine("models_simple/2024-07-29_17-34-34_layer-7-100k/epoch_94_val_loss_0.0896_train_loss_0.0939.pth", 'simple')
  #  medium_engine = Engine("models_medium/2024-07-30_16-47-35_layer-22-100k/epoch_71_val_loss_0.0930_train_loss_0.0665.pth", 'medium')
  #  complex_engine = Engine("models_complex/2024-05-03_17-40-57_layer-76_100k/epoch_61_val_loss_0.1162_train_loss_0.0619.pth", 'complex')
    simple_engine = Engine("models_simple/2024-07-30_12-17-32_layer-7-1M/epoch_97_val_loss_0.2968_train_loss_0.3534.pth", 'simple')
    medium_engine = Engine("models_medium/2024-07-30_17-11-11_layer-22-1M/epoch_99_val_loss_0.2058_train_loss_0.2154.pth", 'medium')
    complex_engine = Engine("models_complex/2024-06-25_17-50-09_layer-76-1M/epoch_47_val_loss_0.2126_train_loss_0.1879.pth", 'complex')

    start_time = time.time()

    # Play games
    results = []
    game_lengths = []
    total_moves = 0
    game_count = 0

    # Track wins
    win_count = {
        'simple_white_vs_medium_black': 0,
        'medium_white_vs_simple_black': 0,
        'medium_white_vs_complex_black': 0,
        'complex_white_vs_medium_black': 0,
        'complex_white_vs_simple_black': 0,
        'simple_white_vs_complex_black': 0
    }

    for _ in range(20):  # Play 50 sets of games
        start_game_time = time.time()
        result, moves = play_game(simple_engine, medium_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('simple', 'medium', result, game_length))
        if "True 1-0" in result:
            win_count['simple_white_vs_medium_black'] += 1
        elif "False 0-1" in result:
            win_count['medium_white_vs_simple_black'] += 1
        print(
            f"Game {game_count}: simple (WHITE) vs medium (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

        start_game_time = time.time()
        result, moves = play_game(medium_engine, simple_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('medium', 'simple', result, game_length))
        if "True 1-0" in result:
            win_count['medium_white_vs_simple_black'] += 1
        elif "False 0-1" in result:
            win_count['simple_white_vs_medium_black'] += 1
        print(
            f"Game {game_count}: medium (WHITE) vs simple (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

        start_game_time = time.time()
        result, moves = play_game(medium_engine, complex_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('medium', 'complex', result, game_length))
        if "True 1-0" in result:
            win_count['medium_white_vs_complex_black'] += 1
        elif "False 0-1" in result:
            win_count['complex_white_vs_medium_black'] += 1
        print(
            f"Game {game_count}: medium (WHITE) vs complex (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

        start_game_time = time.time()
        result, moves = play_game(complex_engine, medium_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('complex', 'medium', result, game_length))
        if "True 1-0" in result:
            win_count['complex_white_vs_medium_black'] += 1
        elif "False 0-1" in result:
            win_count['medium_white_vs_complex_black'] += 1
        print(
            f"Game {game_count}: complex (WHITE) vs medium (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

        start_game_time = time.time()
        result, moves = play_game(complex_engine, simple_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('complex', 'simple', result, game_length))
        if "True 1-0" in result:
            win_count['complex_white_vs_simple_black'] += 1
        elif "False 0-1" in result:
            win_count['simple_white_vs_complex_black'] += 1
        print(
            f"Game {game_count}: complex (WHITE) vs simple (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

        start_game_time = time.time()
        result, moves = play_game(simple_engine, complex_engine)
        game_length = time.time() - start_game_time
        game_lengths.append(game_length)
        total_moves += moves
        game_count += 1
        results.append(('simple', 'complex', result, game_length))
        if "True 1-0" in result:
            win_count['simple_white_vs_complex_black'] += 1
        elif "False 0-1" in result:
            win_count['complex_white_vs_simple_black'] += 1
        print(
            f"Game {game_count}: simple (WHITE) vs complex (BLACK), Result: {result}, Length: {game_length:.2f} seconds")

    end_time = time.time()

    # Calculate average game length and moves
    average_game_length = sum(game_lengths) / len(game_lengths)
    average_moves = total_moves / len(results)

    # Store results
    with open("results/game_results_1M_new_20_2.txt", "w") as f:
        f.write(f"Average game length: {average_game_length:.2f} seconds\n")
        f.write(f"Average number of moves: {average_moves:.2f}\n")
        f.write(f"Total time taken for playing games: {end_time - start_time:.2f} seconds\n\n")

        f.write(f"Win counts:\n")
        f.write(f"Simple (WHITE) vs Medium (BLACK): {win_count['simple_white_vs_medium_black']} wins\n")
        f.write(f"Medium (WHITE) vs Simple (BLACK): {win_count['medium_white_vs_simple_black']} wins\n")
        f.write(f"Medium (WHITE) vs Complex (BLACK): {win_count['medium_white_vs_complex_black']} wins\n")
        f.write(f"Complex (WHITE) vs Medium (BLACK): {win_count['complex_white_vs_medium_black']} wins\n")
        f.write(f"Complex (WHITE) vs Simple (BLACK): {win_count['complex_white_vs_simple_black']} wins\n")
        f.write(f"Simple (WHITE) vs Complex (BLACK): {win_count['simple_white_vs_complex_black']} wins\n\n")

        for match in results:
            f.write(f"{match[0]} (WHITE) vs {match[1]} (BLACK):\n{match[2]}\nLength: {match[3]:.2f} seconds\n\n")

    print("Games completed and results saved.")
    print(f"Program started at: {time.ctime(start_time)}")
    print(f"Program ended at: {time.ctime(end_time)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")