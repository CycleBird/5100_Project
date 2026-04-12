"""
Move pruning utilities for local-window search.

This file is responsible for selecting a smaller candidate move set
from the full board.availables list, so that MCTS only searches within
a local region (for example, a 6x6 window on an 8x8 board).

@author: Zengzheng Jiang
"""

def move_to_rc(move, width):
    """Convert a flat move index to (row, col)."""
    row = move // width
    col = move % width
    return row, col


def rc_to_move(row, col, width):
    """Convert (row, col) back to a flat move index."""
    return row * width + col


def get_window_bounds(board, window_size=6):
    """
    Compute a local window around the center of existing stones.

    Returns:
        row_min, row_max, col_min, col_max
    where row_max and col_max are inclusive.
    """
    width = board.width
    height = board.height

    # If there are no stones yet, use the full board.
    if not board.states:
        return 0, height - 1, 0, width - 1

    moves = list(board.states.keys())
    rows = [move // width for move in moves]
    cols = [move % width for move in moves]

    center_row = int(round(sum(rows) / len(rows)))
    center_col = int(round(sum(cols) / len(cols)))

    half = window_size // 2

    row_min = center_row - half
    col_min = center_col - half

    # Make window_size exactly window_size when possible
    row_max = row_min + window_size - 1
    col_max = col_min + window_size - 1

    # Shift window back inside board if it goes out of bounds
    if row_min < 0:
        row_max += -row_min
        row_min = 0
    if col_min < 0:
        col_max += -col_min
        col_min = 0
    if row_max >= height:
        shift = row_max - height + 1
        row_min -= shift
        row_max = height - 1
    if col_max >= width:
        shift = col_max - width + 1
        col_min -= shift
        col_max = width - 1

    # Final safeguard
    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(height - 1, row_max)
    col_max = min(width - 1, col_max)

    return row_min, row_max, col_min, col_max


def get_pruned_moves(board, window_size=6, fallback_to_full=True):
    """
    Return candidate moves inside a local window.

    Args:
        board: current board object
        window_size: local window size, e.g. 6 for a 6x6 window
        fallback_to_full: if True, fall back to full board.availables
                          when pruning gives an empty result

    Returns:
        A list of legal moves after pruning.
    """
    # If the whole board is already small enough, no need to prune.
    if board.width <= window_size and board.height <= window_size:
        return list(board.availables)

    row_min, row_max, col_min, col_max = get_window_bounds(board, window_size)

    pruned_moves = []
    for move in board.availables:
        row, col = move_to_rc(move, board.width)
        if row_min <= row <= row_max and col_min <= col <= col_max:
            pruned_moves.append(move)

    if not pruned_moves and fallback_to_full:
        return list(board.availables)

    return pruned_moves