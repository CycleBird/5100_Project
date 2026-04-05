# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import argparse
import csv
import json
import os
import random
import time
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
# from policy_value_net import PolicyValueNet  # Theano and Lasagne
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
# from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
# from policy_value_net_keras import PolicyValueNet # Keras


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
            'pure_mcts_playout_num',
            'best_win_ratio',
        ]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    writer.writeheader()

    def write_config(self, config):
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


class TrainPipeline():
    def __init__(self, init_model=None, config=None):
        config = config or {}
        # params of the board and the game
        self.board_width = config.get('board_width', 6)
        self.board_height = config.get('board_height', 6)
        self.n_in_row = config.get('n_in_row', 4)
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = config.get('learn_rate', 2e-3)
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = config.get('temp', 1.0)  # the temperature param
        self.n_playout = config.get('n_playout', 400)  # num of simulations for each move
        self.c_puct = config.get('c_puct', 5)
        self.buffer_size = config.get('buffer_size', 10000)
        self.batch_size = config.get('batch_size', 512)  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = config.get('play_batch_size', 1)
        self.epochs = config.get('epochs', 5)  # num of train_steps for each update
        self.kl_targ = config.get('kl_targ', 0.02)
        self.check_freq = config.get('check_freq', 50)
        self.game_batch_num = config.get('game_batch_num', 1500)
        self.eval_games = config.get('eval_games', 10)
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = config.get('pure_mcts_playout_num', 1000)
        self.output_dir = config.get('output_dir')
        self.device = config.get('device', 'auto')
        self.metrics_logger = MetricsLogger(self.output_dir)
        self.run_started_at = time.time()
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model,
                                                   device=self.device)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   device=self.device)
        print("training device: {}".format(self.policy_value_net.device))
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
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
            'pure_mcts_playout_num': self.pure_mcts_playout_num,
            'output_dir': self.output_dir,
            'device': str(self.policy_value_net.device),
            'init_model': init_model,
        })

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        last_result = {}
        for i in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player,
                                                          temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
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
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
            update_steps = i + 1
        if update_steps == 0:
            update_steps = i + 1
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
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
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win_ratio:{:.3f}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_ratio,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return {
            'win_ratio': win_ratio,
            'eval_wins': win_cnt[1],
            'eval_losses': win_cnt[2],
            'eval_ties': win_cnt[-1],
            'pure_mcts_playout_num': self.pure_mcts_playout_num,
        }

    def build_metrics_row(self, batch_index, selfplay_info):
        row = {
            'timestamp': round(time.time() - self.run_started_at, 3),
            'batch': batch_index,
            'episode_len': selfplay_info.get('episode_len'),
            'augmented_samples': selfplay_info.get('augmented_samples'),
            'buffer_size': len(self.data_buffer),
            'best_win_ratio': self.best_win_ratio,
        }
        row['pure_mcts_playout_num'] = self.pure_mcts_playout_num
        return row

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                selfplay_info = self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.episode_len))
                row = self.build_metrics_row(i + 1, selfplay_info)
                if len(self.data_buffer) > self.batch_size:
                    row.update(self.policy_update())
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    eval_metrics = self.policy_evaluate(self.eval_games)
                    row.update(eval_metrics)
                    self.policy_value_net.save_model('./current_policy.model')
                    if eval_metrics['win_ratio'] > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = eval_metrics['win_ratio']
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                    row['best_win_ratio'] = self.best_win_ratio
                    row['pure_mcts_playout_num'] = self.pure_mcts_playout_num
                self.metrics_logger.log(row)
        except KeyboardInterrupt:
            print('\n\rquit')


def build_parser():
    parser = argparse.ArgumentParser(description='Train AlphaZero Gomoku and export metrics.')
    parser.add_argument('--init-model', default=None, help='Path to an existing model file.')
    parser.add_argument('--board-width', type=int, default=6)
    parser.add_argument('--board-height', type=int, default=6)
    parser.add_argument('--n-in-row', type=int, default=4)
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
    parser.add_argument('--pure-mcts-playout-num', type=int, default=1000)
    parser.add_argument('--output-dir', default='training_artifacts')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    training_pipeline = TrainPipeline(
        init_model=args.init_model,
        config={
            'board_width': args.board_width,
            'board_height': args.board_height,
            'n_in_row': args.n_in_row,
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
            'pure_mcts_playout_num': args.pure_mcts_playout_num,
            'output_dir': args.output_dir,
            'device': args.device,
        }
    )
    training_pipeline.run()
