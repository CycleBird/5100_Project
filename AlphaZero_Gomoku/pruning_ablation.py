"""
Fast pruning ablation for AlphaZero Gomoku.

This script does not train a new model. It loads the provided numpy policy,
plays a small number of low-playout games, and writes CSV files that can be
used for the pruning table and trade-off curve.
"""

from __future__ import print_function

import argparse
import csv
import os
import pickle
import time

import numpy as np

from game import Board, Game
from mcts_alphaZero_pruned import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(path):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(THIS_DIR, path)


def load_numpy_policy(width, height, model_file):
    model_path = resolve_path(model_file)
    try:
        with open(model_path, 'rb') as f:
            policy_param = pickle.load(f)
    except Exception:
        with open(model_path, 'rb') as f:
            policy_param = pickle.load(f, encoding='bytes')
    print("loaded model:", model_path)
    return PolicyValueNetNumpy(width, height, policy_param)


def average(values):
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def parse_variant(token):
    token = token.strip().lower()
    if token == 'full':
        return {
            'name': 'full',
            'window_size': None,
            'policy_top_k': None,
            'include_threats': False,
        }

    include_threats = True
    if token.endswith('plain'):
        include_threats = False
        token = token[:-5]

    if not token.startswith('w'):
        raise ValueError(
            "Variant should look like full, w6plain, w6, or w6k20: " + token
        )

    body = token[1:]
    if 'k' in body:
        window_text, top_k_text = body.split('k', 1)
        policy_top_k = int(top_k_text)
    else:
        window_text = body
        policy_top_k = None

    window_size = int(window_text)
    name = 'window{}'.format(window_size)
    if not include_threats:
        name += '_plain'
    if policy_top_k is not None:
        name += '_top{}'.format(policy_top_k)

    return {
        'name': name,
        'window_size': window_size,
        'policy_top_k': policy_top_k,
        'include_threats': include_threats,
    }


class MeasuredPlayer(object):
    def __init__(self, player):
        self.player_impl = player
        self.player = None
        self.move_count = 0
        self.total_seconds = 0.0
        self.candidate_counts = []

    def set_player_ind(self, p):
        self.player = p
        self.player_impl.set_player_ind(p)

    def reset_player(self):
        self.player_impl.reset_player()

    def get_action(self, board):
        started = time.perf_counter()
        move = self.player_impl.get_action(board)
        elapsed = time.perf_counter() - started

        self.move_count += 1
        self.total_seconds += elapsed
        self.candidate_counts.append(self.player_impl.last_candidate_count)
        return move

    def __str__(self):
        return str(self.player_impl)


def make_player(policy_value_fn, playouts, c_puct, variant):
    return MCTSPlayer(
        policy_value_fn,
        c_puct=c_puct,
        n_playout=playouts,
        is_selfplay=0,
        window_size=variant['window_size'],
        policy_top_k=variant['policy_top_k'],
        include_threats=variant['include_threats'],
    )


def play_variant_games(policy_value_fn, args, variant):
    wins = 0
    losses = 0
    ties = 0
    variant_seconds = []
    baseline_seconds = []
    variant_candidates = []
    baseline_candidates = []

    full_variant = {
        'name': 'full',
        'window_size': None,
        'policy_top_k': None,
        'include_threats': False,
    }

    for game_index in range(args.games):
        board = Board(
            width=args.board_width,
            height=args.board_height,
            n_in_row=args.n_in_row,
        )
        game = Game(board)

        tested_player = MeasuredPlayer(
            make_player(policy_value_fn, args.playouts, args.c_puct, variant)
        )
        full_player = MeasuredPlayer(
            make_player(policy_value_fn, args.playouts, args.c_puct, full_variant)
        )

        if game_index % 2 == 0:
            player1 = tested_player
            player2 = full_player
        else:
            player1 = full_player
            player2 = tested_player

        winner = game.start_play(player1, player2, start_player=0, is_shown=0)

        if winner == -1:
            ties += 1
        elif winner == tested_player.player:
            wins += 1
        else:
            losses += 1

        variant_seconds.append(tested_player.total_seconds)
        baseline_seconds.append(full_player.total_seconds)
        variant_candidates.extend(tested_player.candidate_counts)
        baseline_candidates.extend(full_player.candidate_counts)

        print(
            "{} game {}/{} finished: wins={}, losses={}, ties={}".format(
                variant['name'], game_index + 1, args.games, wins, losses, ties
            )
        )

    win_ratio = (wins + 0.5 * ties) / float(args.games)
    variant_moves = max(1, len(variant_candidates))
    baseline_moves = max(1, len(baseline_candidates))

    return {
        'variant': variant['name'],
        'window_size': 'full' if variant['window_size'] is None else variant['window_size'],
        'policy_top_k': '' if variant['policy_top_k'] is None else variant['policy_top_k'],
        'include_threats': variant['include_threats'],
        'games': args.games,
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_ratio': win_ratio,
        'avg_candidate_moves': average(variant_candidates),
        'baseline_avg_candidate_moves': average(baseline_candidates),
        'seconds': sum(variant_seconds),
        'baseline_seconds': sum(baseline_seconds),
        'seconds_per_move': sum(variant_seconds) / variant_moves,
        'baseline_seconds_per_move': sum(baseline_seconds) / baseline_moves,
    }


def add_tradeoff_numbers(row):
    baseline_candidates = row['baseline_avg_candidate_moves']
    baseline_seconds = row['baseline_seconds_per_move']

    if baseline_candidates > 0:
        row['candidate_reduction'] = 1.0 - row['avg_candidate_moves'] / baseline_candidates
    else:
        row['candidate_reduction'] = 0.0

    if baseline_seconds > 0:
        row['time_reduction'] = 1.0 - row['seconds_per_move'] / baseline_seconds
    else:
        row['time_reduction'] = 0.0

    row['strength_preserved'] = min(1.0, row['win_ratio'] / 0.5)
    return row

# work on the tabling
def write_rows(path, rows):
    fieldnames = [
        'variant',
        'window_size',
        'policy_top_k',
        'include_threats',
        'games',
        'wins',
        'losses',
        'ties',
        'win_ratio',
        'avg_candidate_moves',
        'baseline_avg_candidate_moves',
        'candidate_reduction',
        'seconds',
        'baseline_seconds',
        'seconds_per_move',
        'baseline_seconds_per_move',
        'time_reduction',
        'strength_preserved',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tradeoff(path, rows):
    fieldnames = [
        'variant',
        'candidate_reduction',
        'time_reduction',
        'win_ratio',
        'strength_preserved',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def maybe_write_plot(path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped plot:", path)
        return

    rows = sorted(rows, key=lambda row: row['candidate_reduction'])
    x_values = [row['candidate_reduction'] for row in rows]
    y_values = [row['strength_preserved'] for row in rows]
    labels = [row['variant'] for row in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(x_values, y_values, marker='o')
    for x_value, y_value, label in zip(x_values, y_values, labels):
        plt.annotate(label, (x_value, y_value), textcoords='offset points', xytext=(5, 5))
    plt.xlabel('candidate move reduction')
    plt.ylabel('playing strength preserved')
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("wrote:", path)

# accepts input flags to parse from the CLI
def build_parser():
    parser = argparse.ArgumentParser(description='Run a quick pruning ablation.')
    parser.add_argument('--model-file', default='best_policy_8_8_5.model')
    parser.add_argument('--board-width', type=int, default=8)
    parser.add_argument('--board-height', type=int, default=8)
    parser.add_argument('--n-in-row', type=int, default=5)
    parser.add_argument('--playouts', type=int, default=16)
    parser.add_argument('--games', type=int, default=2)
    parser.add_argument('--c-puct', type=float, default=5)
    parser.add_argument(
        '--variants',
        default='w6plain,w6,w6k20,w5k12',
        help='Comma-separated variants: full, w6plain, w6, w6k20, w5k12, etc.',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', default='pruning_ablation_results')
    parser.add_argument('--plot', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    policy_net = load_numpy_policy(args.board_width, args.board_height, args.model_file)
    variants = [parse_variant(token) for token in args.variants.split(',')]

    rows = []
    for variant in variants:
        row = play_variant_games(policy_net.policy_value_fn, args, variant)
        rows.append(add_tradeoff_numbers(row))

    summary_path = os.path.join(output_dir, 'summary.csv')
    tradeoff_path = os.path.join(output_dir, 'tradeoff.csv')
    write_rows(summary_path, rows)
    write_tradeoff(tradeoff_path, rows)

    print("wrote:", summary_path)
    print("wrote:", tradeoff_path)

    if args.plot:
        plot_path = os.path.join(output_dir, 'tradeoff.png')
        maybe_write_plot(plot_path, rows)


if __name__ == '__main__':
    main()
