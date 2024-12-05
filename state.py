import chess
import numpy as np

class State(object):
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board
        self.move_history = []      #used for saving last played moves

    def serialize_complex(self):
        assert self.board.is_valid()
        bstate = np.zeros((76, 8, 8), np.uint8)

        # Mapping from piece type to layer index
        piece_to_layer = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
        }

        # Initialize piece count layers
        white_piece_counts = {
            chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0,
            chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0
        }
        black_piece_counts = {
            chess.PAWN: 0, chess.KNIGHT: 0, chess.BISHOP: 0,
            chess.ROOK: 0, chess.QUEEN: 0, chess.KING: 0
        }

        # Pieces ---> 0-11
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                layer_index = piece_to_layer[piece.piece_type]
                if piece.color == chess.BLACK:
                    layer_index += 6  # Black pieces are on layers 6-11
                    black_piece_counts[piece.piece_type] += 1
                else:
                    white_piece_counts[piece.piece_type] += 1
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] = 1

        # Castling rights ---> 12
        bstate[12, 0, 0] = self.board.has_queenside_castling_rights(chess.WHITE)
        bstate[12, 0, 7] = self.board.has_kingside_castling_rights(chess.WHITE)
        bstate[12, 7, 0] = self.board.has_queenside_castling_rights(chess.BLACK)
        bstate[12, 7, 7] = self.board.has_kingside_castling_rights(chess.BLACK)

        # En passant square ---> 13
        if self.board.ep_square is not None:
            ep_row, ep_col = divmod(self.board.ep_square, 8)
            bstate[13, ep_row, ep_col] = 1

        # Turn layer ---> 14
        bstate[14, :, :] = self.board.turn

        # Add threat layers
        self.__add_threat_layers(bstate, 15, chess.WHITE)  # Adding threat layer for white
        self.__add_threat_layers(bstate, 16, chess.BLACK)  # Adding threat layer for black

        # Move history ---> 17-33 (2 layers per move, 8 moves total)
        history_idx = 17  # starting index for move history layers
        for i in range(min(8, len(self.move_history))):
            move = self.move_history[-i-1]
            from_square = move.from_square
            to_square = move.to_square
            from_row, from_col = divmod(from_square, 8)
            to_row, to_col = divmod(to_square, 8)
            bstate[history_idx, from_row, from_col] = 1
            bstate[history_idx + 1, to_row, to_col] = 1
            history_idx += 2

        # Piece counts ---> 33-38(white), 39-44(black)
        for piece_type, count in white_piece_counts.items():
            layer_index = 33 + piece_type - 1  # Starting index for white piece counts
            bstate[layer_index, 0, :] = count  # Use the first row to record counts

        for piece_type, count in black_piece_counts.items():
            layer_index = 39 + piece_type - 1  # Starting index for black piece counts
            bstate[layer_index, 0, :] = count  # Use the first row to record counts

        # King Safety ---> 45-46
        white_king_square = self.board.king(chess.WHITE)
        black_king_square = self.board.king(chess.BLACK)
        self.__add_king_safety_layer(bstate, white_king_square, 45)  # Layer 45 for white king safety
        self.__add_king_safety_layer(bstate, black_king_square, 46)  # Layer 46 for black king safety

        # Control of center ---> 47-48
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for square in center_squares:
            if self.board.is_attacked_by(chess.WHITE, square):
                row, col = divmod(square, 8)
                bstate[47, row, col] += 1  # Incremental control for white
            if self.board.is_attacked_by(chess.BLACK, square):
                row, col = divmod(square, 8)
                bstate[48, row, col] += 1  # Incremental control for black

        # Pressure Map
        self.__add_pressure_map(bstate, 49, chess.WHITE)  # Layer 49 for white pressure
        self.__add_pressure_map(bstate, 50, chess.BLACK)  # Layer 50 for black pressure

        # Pins
        self.__add_pin_layer(bstate, 51, chess.WHITE)  # Layer 51 for white pins
        self.__add_pin_layer(bstate, 52, chess.BLACK)  # Layer 52 for black pins

        # Adding pawn structure analysis layers
        self.__add_pawn_structure_layers(bstate, 53, chess.WHITE)  # Layers 53-55 for white
        self.__add_pawn_structure_layers(bstate, 56, chess.BLACK)  # Layers 56-58 for black

        # Adding mobility analysis layers
        self.__add_mobility_layers(bstate, 59, chess.WHITE)  # Layers 59-64 for white
        self.__add_mobility_layers(bstate, 64, chess.BLACK)  # Layers 65-70 for black

        # Adding control value layers
        self.__add_control_value_layers(bstate, 71, chess.WHITE)  # Layer 71 for white control values
        self.__add_control_value_layers(bstate, 72, chess.BLACK)  # Layer 72 for black control values

        # Adding look-ahead feature layers
        self.__add_look_ahead_layers(bstate, 73)  # Layer 73 for white's look-ahead, 74 for black's look-ahead

        # Adding game phase layer
        self.__add_game_phase_layer(bstate, 75)  # Layer 75 for game phase

        return bstate

    def serialize_medium(self):
        assert self.board.is_valid()
        bstate = np.zeros((22, 8, 8), np.uint8)

        # Mapping from piece type to layer index
        piece_to_layer = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
        }

        # Pieces ---> 0-11
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                layer_index = piece_to_layer[piece.piece_type]
                if piece.color == chess.BLACK:
                    layer_index += 6  # Black pieces are on layers 6-11
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] = 1

        # Castling rights ---> 12
        bstate[12, 0, 0] = self.board.has_queenside_castling_rights(chess.WHITE)
        bstate[12, 0, 7] = self.board.has_kingside_castling_rights(chess.WHITE)
        bstate[12, 7, 0] = self.board.has_queenside_castling_rights(chess.BLACK)
        bstate[12, 7, 7] = self.board.has_kingside_castling_rights(chess.BLACK)

        # En passant square ---> 13
        if self.board.ep_square is not None:
            ep_row, ep_col = divmod(self.board.ep_square, 8)
            bstate[13, ep_row, ep_col] = 1

        # Turn layer ---> 14
        bstate[14, :, :] = self.board.turn

        # Threats layers
        self.__add_threat_layers(bstate, 15, chess.WHITE)  # Adding threat layer for white
        self.__add_threat_layers(bstate, 16, chess.BLACK)  # Adding threat layer for black

        # Move history ---> 17-21 (2 layers per move, 2 moves total)
        history_idx = 17  # starting index for move history layers
        for i in range(min(2, len(self.move_history))):
            move = self.move_history[-i - 1]
            from_square = move.from_square
            to_square = move.to_square
            from_row, from_col = divmod(from_square, 8)
            to_row, to_col = divmod(to_square, 8)
            bstate[history_idx, from_row, from_col] = 1
            bstate[history_idx + 1, to_row, to_col] = 1
            history_idx += 2

        return bstate

    def serialize_simple(self):
        assert self.board.is_valid()
        bstate = np.zeros((7, 8, 8), np.uint8)

        # Mapping from piece type to layer index
        piece_to_layer = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
        }

        # Pieces ---> 0-5
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                layer_index = piece_to_layer[piece.piece_type]
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] = 1

        # Turn layer ---> 6
        bstate[6, :, :] = self.board.turn

        return bstate

    def make_move(self, move):   #used in engine and server
        try:
            move = self.board.parse_san(move) if isinstance(move, str) else move
            if move in self.board.legal_moves:
                self.board.push(move)
                return True
            else:
                return False
        except ValueError:
            return False

    def is_game_over(self):
    #    print("Checking game over conditions...")
    #    print(f"Current board:\n{self.board}")

        # Check for checkmate
        if self.board.is_checkmate():
            winner = "Black" if self.board.turn == chess.WHITE else "White"
          #  print(f"Game over by checkmate. {winner} wins.")
          #  print(self.board)
            return True

        # Check for stalemate
        if self.board.is_stalemate():
          #  print("Game over by stalemate.")
          #  print(self.board)
            return True

        # Check for insufficient material
        if self.board.is_insufficient_material():
          #  print("Game over by insufficient material.")
          #  print(self.board)
            return True

        # Check for other draw conditions not always included by default:
    #    if self.board.can_claim_draw():
         #   print("Game over by draw claim.")
          #  print(self.board)
     #       return True

     #   print("No game-ending condition met.")
        return False

    def legal_moves(self):
        return list(self.board.legal_moves)

    def reset(self):
        self.board.reset()

    def __add_threat_layers(self, bstate, layer_index, color):
        """Adds a layer indicating the threat levels for each square by the given color directly to bstate."""
        for square in chess.SQUARES:
            if self.board.is_attacked_by(color, square):
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] += 1

    def __add_king_safety_layer(self, bstate, king_square, layer_index):
        """Adds information about the safety of the king to the given layer index."""
        row, col = divmod(king_square, 8)
        # Mark surrounding squares to indicate the region to evaluate for safety
        for drow in [-1, 0, 1]:
            for dcol in [-1, 0, 1]:
                nrow, ncol = row + drow, col + dcol
                if 0 <= nrow < 8 and 0 <= ncol < 8:
                    bstate[layer_index, nrow, ncol] = 1

    def __add_pressure_map(self, bstate, layer_index, color):
        """Adds a layer indicating the number of attacks and defenses on each square."""
        for square in chess.SQUARES:
            attackers = len(self.board.attackers(color, square))
            if attackers > 0:
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] = attackers

    def __add_pin_layer(self, bstate, layer_index, color):
        """Adds a layer marking pinned pieces against the king."""
        for square in chess.SQUARES:
            if self.board.is_pinned(color, square):
                row, col = divmod(square, 8)
                bstate[layer_index, row, col] = 1

    def __add_pawn_structure_layers(self, bstate, start_layer, color):
        """Adds layers identifying doubled, isolated, and backward pawns."""
        file_pawns = [0] * 8  # Track the number of pawns in each file
        pawn_positions = []  # List of pawn positions

        # Calculate pawn counts per file and collect pawn positions
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                file_pawns[file] += 1
                pawn_positions.append((rank, file))

        # Determine doubled, isolated, and backward pawns
        for rank, file in pawn_positions:
            row, col = 7 - rank, file
            # Doubled pawns
            if file_pawns[file] > 1:
                bstate[start_layer, row, col] = 1
            # Isolated pawns
            if (file == 0 or file_pawns[file - 1] == 0) and (file == 7 or file_pawns[file + 1] == 0):
                bstate[start_layer + 1, row, col] = 1
            # Backward pawns (simplified version)
            if (file > 0 and rank > max([r for r, f in pawn_positions if f == file - 1], default=-1)) or \
                    (file < 7 and rank > max([r for r, f in pawn_positions if f == file + 1], default=-1)):
                bstate[start_layer + 2, row, col] = 1

    def __add_mobility_layers(self, bstate, start_layer, color):
        """Adds layers showing the mobility for each piece type."""
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            mobility_map = np.zeros((8, 8), np.uint8)
            for square in chess.SQUARES:
                piece = self.board.piece_at(square)
                if piece is not None and piece.piece_type == piece_type and piece.color == color:
                    legal_moves = self.board.generate_legal_moves(from_mask=chess.BB_SQUARES[square])
                    mobility_count = sum(1 for _ in legal_moves)
                    row, col = divmod(square, 8)
                    mobility_map[row, col] = mobility_count
            layer_index = start_layer + piece_type - 1
            bstate[layer_index] = mobility_map

    def __add_control_value_layers(self, bstate, layer_index, color):
        """Adds a layer indicating the weighted control value for each square."""
        piece_value = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        control_map = np.zeros((8, 8), np.uint8)

        for square in chess.SQUARES:
            attackers = self.board.attackers(color, square)
            value_sum = sum(piece_value[self.board.piece_at(attacker).piece_type] for attacker in attackers)
            row, col = divmod(square, 8)
            control_map[row, col] = value_sum

        bstate[layer_index] = control_map

    def __add_look_ahead_layers(self, bstate, start_layer):
        """Adds layers estimating the strength of future positions after potential moves."""
        for color in [chess.WHITE, chess.BLACK]:
            look_ahead_map = np.zeros((8, 8), np.uint8)
            # Temporarily make moves and evaluate the position
            for move in self.board.legal_moves:
                if self.board.color_at(move.from_square) == color:
                    self.board.push(move)
                    # A simple evaluation function: difference in material
                    score = self.__evaluate_position()
                    row, col = divmod(move.to_square, 8)
                    look_ahead_map[row, col] = max(look_ahead_map[row, col], score)
                    self.board.pop()

            layer_index = start_layer + (0 if color == chess.WHITE else 1)
            bstate[layer_index] = look_ahead_map

    def __evaluate_position(self, move=None):
        """Simple position evaluation counting material, with bonuses for captures."""
        value = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9}
        score = 0
        capture_bonus = 10  # Bonus for capturing pieces

        # Apply the move if provided
        if move:
            piece_captured = self.board.piece_at(move.to_square)
            if piece_captured:
                piece_value = value.get(piece_captured.symbol().lower(), 0)
                if piece_captured.color == chess.WHITE:
                    score -= piece_value + capture_bonus
                else:
                    score += piece_value + capture_bonus

        # Count the material
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_value = value.get(piece.symbol().lower(), 0)
                if piece.color == chess.WHITE:
                    score += piece_value
                else:
                    score -= piece_value

        return score

    def __add_game_phase_layer(self, bstate, layer_index):
        """Adds a layer indicating the game phase."""
        # Count pieces to determine the game phase
        phase_score = 0
        piece_value = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                phase_score += piece_value.get(piece.symbol().lower(), 0)

        # Heuristic thresholds to determine the game phase
        if phase_score > 70:
            game_phase = 0  # Opening
        elif phase_score > 30:
            game_phase = 1  # Middlegame
        else:
            game_phase = 2  # Endgame

        bstate[layer_index, :, :] = game_phase