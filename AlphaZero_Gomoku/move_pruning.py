"""
Move pruning utilities for local-window search.

First version:
This file is responsible for selecting a smaller candidate move set
from the full board.availables list, so that MCTS only searches within
a local region (for example, a 6x6 window on an 8x8 board).

New things:
The first version only kept legal moves inside a fixed local window.
This version still starts from that simple idea, then adds two small
exceptions that are cheap enough to use in quick experiments:
1. keep immediate winning moves;
2. keep immediate blocking moves against the opponent.

Policy top-k filtering is also kept here so the MCTS code can stay simple.

@author: Zengzheng Jiang, Weizhi Du
"""

def move_to_rc(move, width):
    """Convert a flat move index to (row, col)."""
    row = move // width
    col = move % width
    return row, col


def rc_to_move(row, col, width):
    """Convert (row, col) back to a flat move index."""
    return row * width + col


def _other_player(board, player):
    """Return the opponent id for the two-player Gomoku board."""
    if player == board.players[0]:
        return board.players[1]
    return board.players[0]


def _count_stones(board, row, col, row_step, col_step, player):
    """Count same-player stones from one square in one direction."""
    count = 0
    row += row_step
    col += col_step
    while 0 <= row < board.height and 0 <= col < board.width:
        move = rc_to_move(row, col, board.width)
        if board.states.get(move) != player:
            break
        count += 1
        row += row_step
        col += col_step
    return count


def would_win_after_move(board, move, player):
    """
    Check whether a legal move immediately completes n_in_row for player.

    This only checks the four lines passing through the candidate move, so it is
    much cheaper than running a full board winner scan for every legal move.
    """
    if move not in board.availables:
        return False

    row, col = move_to_rc(move, board.width)
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for row_step, col_step in directions:
        total = 1
        total += _count_stones(board, row, col, row_step, col_step, player)
        total += _count_stones(board, row, col, -row_step, -col_step, player)
        if total >= board.n_in_row:
            return True
    return False


def get_threat_moves(board):
    """
    Return legal moves that should survive pruning.

    A move is threat-critical when it wins immediately for the current player
    or blocks an immediate win by the opponent.
    """
    threat_moves = []
    current_player = board.get_current_player()
    opponent = _other_player(board, current_player)

    for move in board.availables:
        if would_win_after_move(board, move, current_player):
            threat_moves.append(move)
        elif would_win_after_move(board, move, opponent):
            threat_moves.append(move)

    return threat_moves


def get_window_bounds(board, window_size=6):
    """
    Compute a local window around the center of existing stones.

    Returns:
        row_min, row_max, col_min, col_max
    where row_max and col_max are inclusive.
    """
    width = board.width
    height = board.height

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

    row_max = row_min + window_size - 1
    col_max = col_min + window_size - 1

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

    row_min = max(0, row_min)
    col_min = max(0, col_min)
    row_max = min(height - 1, row_max)
    col_max = min(width - 1, col_max)

    return row_min, row_max, col_min, col_max


def get_pruned_moves(board, window_size=6, fallback_to_full=True,
                     include_threats=True):
    """
    Return candidate moves inside a local window.

    Args:
        board: current board object
        window_size: local window size, e.g. 6 for a 6x6 window.
                     Use None to disable window pruning.
        fallback_to_full: if True, fall back to full board.availables
                          when pruning gives an empty result
        include_threats: if True, keep immediate winning and blocking moves
                         even if they fall outside the local window

    Returns:
        A list of legal moves after pruning.
    """
    if window_size is None:
        return list(board.availables)

    if board.width <= window_size and board.height <= window_size:
        return list(board.availables)

    row_min, row_max, col_min, col_max = get_window_bounds(board, window_size)

    pruned_moves = []
    pruned_set = set()
    for move in board.availables:
        row, col = move_to_rc(move, board.width)
        if row_min <= row <= row_max and col_min <= col <= col_max:
            pruned_moves.append(move)
            pruned_set.add(move)

    if include_threats:
        for move in get_threat_moves(board):
            if move not in pruned_set:
                pruned_moves.append(move)
                pruned_set.add(move)

    if not pruned_moves and fallback_to_full:
        return list(board.availables)

    return pruned_moves


def filter_action_probs(action_probs, allowed_moves, policy_top_k=None,
                        keep_moves=None, fallback_to_full=True):
    """
    Filter policy priors to a legal candidate set, then optionally keep top-k.

    keep_moves are not counted against policy_top_k. This is how threat moves
    stay available even when their network prior is low.
    """
    action_probs = list(action_probs)
    allowed_set = set(allowed_moves)
    keep_set = set(keep_moves or [])

    filtered = []

    for act, prob in action_probs:
        if act in allowed_set or act in keep_set:
            filtered.append((act, prob))

    if not filtered and fallback_to_full:
        filtered = action_probs

    if policy_top_k is not None and policy_top_k > 0:
        selected = []
        selected_actions = set()

        for act, prob in filtered:
            if act in keep_set:
                selected.append((act, prob))
                selected_actions.add(act)

        rankable = []

        for act, prob in filtered:
            if act not in selected_actions:
                rankable.append((act, prob))

        ranked = sorted(
            rankable,
            key=lambda item: item[1],
            reverse=True
        )

        count = 0

        for act, prob in ranked:
            if count >= policy_top_k:
                break

            selected.append((act, prob))
            count += 1

        filtered = selected

    prob_sum = 0

    for act, prob in filtered:
        prob_sum += prob

    if prob_sum > 0:
        normalized = []

        for act, prob in filtered:
            normalized.append((act, prob / prob_sum))

        filtered = normalized

    return filtered