from flask import Flask, request, jsonify, url_for
from engine import Engine, computer_move_server
from state import State

app = Flask(__name__, static_folder='gui/static')

# Initialize the chess engine and game state
#valuator = Engine("models_complex/2024-05-03_17-40-57_layer-76_100k/epoch_61_val_loss_0.1162_train_loss_0.0619.pth")
#valuator = Engine("models_complex/2024-06-22_17-58-55_layer-76-200k/epoch_73_val_loss_0.1284_train_loss_0.0778.pth")
valuator = Engine("models_complex/2024-06-25_17-50-09_layer-76-1M/epoch_47_val_loss_0.2126_train_loss_0.1879.pth", 'complex')
state = State()


@app.route("/")
def index():
    """Serve the main page with the chessboard."""
    base_url = url_for('static', filename='img/pieces/')
    return f"""
    <html>
    <head>
        <link rel="stylesheet" href="{url_for('static', filename='css/chessboard.css')}" type="text/css">
        <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" integrity="sha384-q94+BZtLrkL1/ohfjR8c6L+A6qzNH9R2hBLwyoAfu3i/WCvQjzL2RQJ3uNHDISdU" crossorigin="anonymous">
        <script src="https://code.jquery.com/jquery-3.5.1.min.js" integrity="sha384-ZvpUoO/+PpLXR1lu4jmpXWu80pZlYUAfxl5NsBMWOEPSjUn/6Z/hRTt8+pR6L4N2" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/chess.js/0.10.2/chess.min.js"></script>
        <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js" crossorigin="anonymous"></script>
        <script>
            var basePath = '{base_url}';
        </script>
        <script src="{url_for('static', filename='js/chessboard-setup.js')}"></script>
    </head>
    <body>
        <div style="display: flex; flex-direction: column; align-items: center;">
            <div id="result" style="margin-bottom: 10px; color: red; font-weight: bold; font-size: 30px;"></div>
            <div style="display: flex;">
                <div id="board" style="width: 500px"></div>
                <div style="overflow-y: scroll; max-height: 500px; padding-left: 10px; width: 300px;">
                    <h2>Best Moves</h2>
                    <ul id="best-moves">
                        <!-- Best moves will be populated dynamically -->
                    </ul>
                </div>
            </div>
            <div style="margin-top: 20px;">
                <button onclick="resetBoard()">Reset Board</button>
            </div>
        </div>
    </body>
    </html>
    """



@app.route("/move")
def move():
    """Validate a move and return the new board position."""
    source = request.args.get('source')
    target = request.args.get('target')
    move = f'{source}{target}'

    try:
        if state.make_move(move):
            moves_data = computer_move_server(state, valuator)
            best_moves_html = ''.join([f"<li>{move[1]} ({move[0]:.6f})</li>" for move in moves_data[:3]])
            return jsonify(valid=True, newPos=state.board.fen(), best_moves=best_moves_html)
        else:
            return jsonify(valid=False, error="Invalid move", newPos=state.board.fen())
    except Exception as e:
        is_checkmate = state.board.is_checkmate()
        return jsonify(valid=False, error=str(e), newPos=state.board.fen(), checkmate=is_checkmate)


@app.route("/reset", methods=['POST'])
def reset_board():
    global state
    state.reset()
    return jsonify({'message': 'Board reset successfully'})


if __name__ == "__main__":
    app.run(debug=True)
