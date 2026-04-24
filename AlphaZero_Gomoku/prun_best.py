"""
Pruning-based training pipeline for AlphaZero Gomoku
Rule: 8x8 board, 5 in a row
Train for 300 batches
Evaluate every 50 batches against provided pretrained 8x8x5 model

@author: Zengzheng Jiang
"""

from __future__ import print_function
import argparse
import csv
import json
import os
import random
import time
import pickle
import numpy as np
from collections import defaultdict, deque

from game import Board, Game
from mcts_alphaZero_pruned import MCTSPlayer as PrunedMCTSPlayer
from mcts_alphaZero import MCTSPlayer as StandardMCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from policy_value_net_numpy import PolicyValueNetNumpy

from datetime import datetime

class MetricsLogger(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.metrics_path = os.path.join(output_dir, 'metrics.csv')
        self.fieldnames = [
            'timestamp',
            'batch',
            'episode_len',
            'augmented_samples',
            'buffer_size',
            'loss',
            'entropy',
            'kl',
            'lr_multiplier',
            'effective_lr',
            'explained_var_old',
            'explained_var_new',
            'update_steps',
            'win_ratio',
            'eval_wins',
            'eval_losses',
            'eval_ties',
            'best_win_ratio',
            'opponent_model_file',
        ]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()

    def write_config(self, config):
        if not self.output_dir:
            return
        config_path = os.path.join(self.output_dir, 'run_config.json')
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=2, sort_keys=True)

    def log(self, row):
        if not self.output_dir:
            return
        normalized_row = {field: row.get(field, '') for field in self.fieldnames}
        with open(self.metrics_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(normalized_row)


def load_pretrained_numpy_policy(width, height, model_candidates):
    """
    Load provided 8x8x5 pretrained model saved in pickle format.
    Try .model first, then .model2 as fallback.
    """
    last_error = None
    for model_file in model_candidates:
        if not os.path.exists(model_file):
            continue
        try:
            with open(model_file, 'rb') as f:
                policy_param = pickle.load(f)
            print("Loaded pretrained opponent model from:", model_file)
            return PolicyValueNetNumpy(width, height, policy_param), model_file
        except Exception as e1:
            last_error = e1
            try:
                with open(model_file, 'rb') as f:
                    policy_param = pickle.load(f, encoding='bytes')
                print("Loaded pretrained opponent model from:", model_file)
                return PolicyValueNetNumpy(width, height, policy_param), model_file
            except Exception as e2:
                last_error = e2

    raise RuntimeError(
        "Could not load any pretrained opponent model from {}. Last error: {}".format(
            model_candidates, last_error
        )
    )


class PruningTrainPipeline(object):
    def __init__(self, init_model=None, config=None):
        config = config or {}

        self.board_width = config.get('board_width', 8)
        self.board_height = config.get('board_height', 8)
        self.n_in_row = config.get('n_in_row', 5)

        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row
        )
        self.game = Game(self.board)

        self.learn_rate = config.get('learn_rate', 2e-3)
        self.lr_multiplier = 1.0
        self.temp = config.get('temp', 1.0)
        self.n_playout = config.get('n_playout', 400)
        self.c_puct = config.get('c_puct', 5)
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 512)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = config.get('play_batch_size', 1)
        self.epochs = config.get('epochs', 5)
        self.kl_targ = config.get('kl_targ', 0.02)
        self.check_freq = config.get('check_freq', 50)
        self.game_batch_num = config.get('game_batch_num', 1500)
        self.eval_games = config.get('eval_games', 10)
        self.best_win_ratio = 0.0
        self.output_dir = config.get('output_dir', 'pruning_training_artifacts')

        self.metrics_logger = MetricsLogger(self.output_dir)
        self.run_started_at = time.time()

        if init_model:
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model
            )
        else:
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height
            )

        self.mcts_player = PrunedMCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1
        )

        candidate_models = config.get(
            'opponent_model_candidates',
            ['best_policy_8_8_5.model', 'best_policy_8_8_5.model2']
        )
        self.opponent_policy, self.opponent_model_file = load_pretrained_numpy_policy(
            self.board_width,
            self.board_height,
            candidate_models
        )

        self.metrics_logger.write_config({
            'board_width': self.board_width,
            'board_height': self.board_height,
            'n_in_row': self.n_in_row,
            'learn_rate': self.learn_rate,
            'temp': self.temp,
            'n_playout': self.n_playout,
            'c_puct': self.c_puct,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'play_batch_size': self.play_batch_size,
            'epochs': self.epochs,
            'kl_targ': self.kl_targ,
            'check_freq': self.check_freq,
            'game_batch_num': self.game_batch_num,
            'eval_games': self.eval_games,
            'output_dir': self.output_dir,
            'init_model': init_model,
            'opponent_model_file': self.opponent_model_file,
        })

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping"""
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(
                    np.flipud(mcts_prob.reshape(self.board_height, self.board_width)),
                    i
                )
                extend_data.append((
                    equi_state,
                    np.flipud(equi_mcts_prob).flatten(),
                    winner
                ))

                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((
                    equi_state,
                    np.flipud(equi_mcts_prob).flatten(),
                    winner
                ))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        last_result = {}
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(
                self.mcts_player,
                temp=self.temp
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

            last_result = {
                'winner': winner,
                'episode_len': self.episode_len,
                'augmented_samples': len(play_data),
            }
        return last_result

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        kl = 0.0
        loss = None
        entropy = None
        update_steps = 0

        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(
                old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)
                ),
                axis=1
            ))
            if kl > self.kl_targ * 4:
                break
            update_steps = i + 1

        if update_steps == 0:
            update_steps = i + 1

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        winner_var = np.var(np.array(winner_batch))
        if winner_var == 0:
            explained_var_old = 0.0
            explained_var_new = 0.0
        else:
            explained_var_old = (
                1 - np.var(np.array(winner_batch) - old_v.flatten()) / winner_var
            )
            explained_var_new = (
                1 - np.var(np.array(winner_batch) - new_v.flatten()) / winner_var
            )

        print((
            "kl:{:.5f},"
            "lr_multiplier:{:.3f},"
            "loss:{},"
            "entropy:{},"
            "explained_var_old:{:.3f},"
            "explained_var_new:{:.3f}"
        ).format(
            kl,
            self.lr_multiplier,
            loss,
            entropy,
            explained_var_old,
            explained_var_new
        ))

        return {
            'loss': loss,
            'entropy': entropy,
            'kl': kl,
            'lr_multiplier': self.lr_multiplier,
            'effective_lr': self.learn_rate * self.lr_multiplier,
            'explained_var_old': explained_var_old,
            'explained_var_new': explained_var_new,
            'update_steps': update_steps,
        }

    def policy_evaluate(self, n_games=10):
        """
        Evaluate current pruning-trained model against provided pretrained 8x8x5 model.
        Current player uses pruning MCTS.
        Opponent uses standard MCTS + provided pretrained policy.
        """
        current_player = PrunedMCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=0
        )

        opponent_player = StandardMCTSPlayer(
            self.opponent_policy.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=0
        )

        eval_board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row
        )
        eval_game = Game(eval_board)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = eval_game.start_play(
                current_player,
                opponent_player,
                start_player=i % 2,
                is_shown=0
            )
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print(
            "vs pretrained {} | win_ratio:{:.3f}, win:{}, lose:{}, tie:{}".format(
                self.opponent_model_file,
                win_ratio,
                win_cnt[1],
                win_cnt[2],
                win_cnt[-1]
            )
        )

        return {
            'win_ratio': win_ratio,
            'eval_wins': win_cnt[1],
            'eval_losses': win_cnt[2],
            'eval_ties': win_cnt[-1],
            'opponent_model_file': self.opponent_model_file,
        }

    def build_metrics_row(self, batch_index, selfplay_info):
        return {
            'timestamp': round(time.time() - self.run_started_at, 3),
            'batch': batch_index,
            'episode_len': selfplay_info.get('episode_len'),
            'augmented_samples': selfplay_info.get('augmented_samples'),
            'buffer_size': len(self.data_buffer),
            'best_win_ratio': self.best_win_ratio,
            'opponent_model_file': self.opponent_model_file,
        }

    def run(self):
        """run the pruning training pipeline"""
        start_time = datetime.now()
        print("Training start time:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

        try:
            for i in range(self.game_batch_num):
                selfplay_info = self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))

                row = self.build_metrics_row(i + 1, selfplay_info)

                if len(self.data_buffer) > self.batch_size:
                    row.update(self.policy_update())

                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    eval_metrics = self.policy_evaluate(self.eval_games)
                    row.update(eval_metrics)

                    current_model_path = os.path.join(
                        self.output_dir,
                        'pruned_current_policy_8_8_5.model'
                    )
                    best_model_path = os.path.join(
                        self.output_dir,
                        'pruned_best_policy_8_8_5.model'
                    )

                    self.policy_value_net.save_model(current_model_path)

                    if eval_metrics['win_ratio'] > self.best_win_ratio:
                        print("New best pruning policy!!!!!!!!")
                        self.best_win_ratio = eval_metrics['win_ratio']
                        self.policy_value_net.save_model(best_model_path)

                    row['best_win_ratio'] = self.best_win_ratio

                self.metrics_logger.log(row)
            end_time = datetime.now()
            print("Training end time:", end_time.strftime("%Y-%m-%d %H:%M:%S"))

            duration = end_time - start_time
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            print("Total duration: {} hours {} minutes {} seconds".format(
                hours, minutes, seconds
            ))
            
        except KeyboardInterrupt:
            print('\n\rquit')


def build_parser():
    parser = argparse.ArgumentParser(
        description='Train pruning-based AlphaZero Gomoku on 8x8x5 and evaluate vs pretrained 8x8x5 model.'
    )
    parser.add_argument('--init-model', default=None, help='Path to an existing PyTorch model file.')
    parser.add_argument('--learn-rate', type=float, default=2e-3)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--n-playout', type=int, default=400)
    parser.add_argument('--c-puct', type=float, default=5)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--play-batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--kl-targ', type=float, default=0.02)
    parser.add_argument('--check-freq', type=int, default=50)
    parser.add_argument('--game-batch-num', type=int, default=1500)
    parser.add_argument('--eval-games', type=int, default=10)
    parser.add_argument('--output-dir', default='pruning_training_artifacts')
    parser.add_argument(
        '--opponent-models',
        nargs='*',
        default=['best_policy_8_8_5.model', 'best_policy_8_8_5.model2'],
        help='Candidate pretrained opponent model files, tried in order.'
    )
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()

    training_pipeline = PruningTrainPipeline(
        init_model=args.init_model,
        config={
            'board_width': 8,
            'board_height': 8,
            'n_in_row': 5,
            'learn_rate': args.learn_rate,
            'temp': args.temp,
            'n_playout': args.n_playout,
            'c_puct': args.c_puct,
            'buffer_size': args.buffer_size,
            'batch_size': args.batch_size,
            'play_batch_size': args.play_batch_size,
            'epochs': args.epochs,
            'kl_targ': args.kl_targ,
            'check_freq': args.check_freq,
            'game_batch_num': args.game_batch_num,
            'eval_games': args.eval_games,
            'output_dir': args.output_dir,
            'opponent_model_candidates': args.opponent_models,
        }
    )
    training_pipeline.run()