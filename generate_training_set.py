import os
import chess
import chess.pgn
import numpy as np
from state import State


def get_dataset(num_of_samples=None):
    X, Y = [], []
    gn = 0
    rv = {'1/2-1/2': 0, '0-1': -1, '1-0': 1}
    for fn in os.listdir("data"):
        with open(os.path.join("data", fn), 'r') as pgn:
            print("---------------------------------------------", pgn.name, "---------------------------------------------")
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                res = game.headers['Result']
                if res not in rv:
                    continue
                board = game.board()
                for move in game.mainline_moves():
                  #  serialized_board = State(board).serialize_complex()
                  #  serialized_board = State(board).serialize_medium()
                    serialized_board = State(board).serialize_simple()
                    X.append(serialized_board)
                    Y.append(rv[res])
                  #  print(serialized_board, rv[res])
                #    print("Game %d, parsing move %d" % (gn, len(X)))
                    if len(X) % 1000 == 0:
                        print("Game %d, parsing move %d" % (gn, len(X)))
                    board.push(move)
                if num_of_samples is not None and len(X) >= num_of_samples:
                    break
                gn += 1
        if num_of_samples is not None and len(X) >= num_of_samples:
            break
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = get_dataset(100000)
   # np.savez("processed_complex/dataset_1M_layer_76.npz", X=X, Y=Y)
    np.savez("processed_simple/dataset_100k_layer_7.npz", X=X, Y=Y)
