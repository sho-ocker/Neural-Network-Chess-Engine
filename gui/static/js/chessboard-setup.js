document.addEventListener('DOMContentLoaded', function () {
    var game = new Chess();
    var whiteSquareGrey = '#a9a9a9';
    var blackSquareGrey = '#696969';
    var board = Chessboard('board', {
        draggable: true,
        dropOffBoard: 'snapback',
        position: 'start',
        pieceTheme: function(piece) {
            return basePath + piece + '.png';
        },
        onDragStart: onDragStart,
        onDrop: onDrop,
        onMouseoutSquare: onMouseoutSquare,
        onMouseoverSquare: onMouseoverSquare,
        onSnapEnd: onSnapEnd
    });
    var moveHistory = []; // Define moveHistory array to store move history
    var resultDiv = $('#result');

    function handleMove(source, target, piece, newPos, oldPos, orientation) {
        $.getJSON('/move', {source: source, target: target, piece: piece}, function(response) {
            if (response.checkmate) {
                resultDiv.text('Checkmate! The game is over.');
                game.game_over(true);
                board.position(game.fen());
                return;
            }
            else if (response.valid) {
                game.load(response.newPos);
                board.position(game.fen());

                moveHistory.push(response); // Store entire response object
                updateBestMoves(); // Update best moves list
            } else {
                alert('Invalid move: ' + response.error);
                board.position(oldPos);
            }
            console.log(response);
        }).fail(function() {
            alert('Failed to make the move. Check connection and console for errors.');
        });
    }

    window.resetBoard = function() {
        $.ajax({
            url: '/reset',
            method: 'POST',
            success: function(response) {
                game.reset();
                board.position('start', true);  // Reset the board to the initial position
                moveHistory = []; // Reset move history
                updateBestMoves(); // Clear best moves list
                $('#result').text(''); // Clear result message
                console.log('Board and game state reset.');
            },
            error: function() {
                alert('Error resetting the board.');
            }
        });
    };

    function onDragStart(source, piece) {
        if (game.game_over() || (game.turn() === 'w' && piece.search(/^b/) !== -1) ||
            (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
            return false;
        }
    }

    function onDrop(source, target) {
        removeGreySquares();

        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' // NOTE: always promote to a queen for example simplicity
        });

        if (move === null){
            return 'snapback'; // illegal move
        }

        handleMove(source, target);
    }

    function onMouseoverSquare(square, piece) {
        var moves = game.moves({
            square: square,
            verbose: true
        });
        if (moves.length === 0) return;
        greySquare(square);
        for (var i = 0; i < moves.length; i++) {
            greySquare(moves[i].to);
        }
    }

    function onMouseoutSquare(square, piece) {
        removeGreySquares();
    }

    function onSnapEnd() {
        board.position(game.fen());
    }

    function greySquare(square) {
        var $square = $('#board .square-' + square);
        var background = whiteSquareGrey;
        if ($square.hasClass('black-3c85d')) {
            background = blackSquareGrey;
        }
        $square.css('background', background);
    }

    function removeGreySquares() {
        $('#board .square-55d63').css('background', '');
    }

    function updateBestMoves() {
        var bestMovesHtml = '';
        for (var i = 0; i < moveHistory.length; i++) {
            var movesData = moveHistory[i].best_moves;
            bestMovesHtml += '<li>Move ' + (i + 1) + ': </li>';
            bestMovesHtml += movesData;
            bestMovesHtml += '<hr>';
        }
        $('#best-moves').html(bestMovesHtml);
    }
});