# -*- coding: utf-8 -*- 
# @Date : 2023/2/12
# @Author : YEY
# @File : blackjack_offline.py

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from collections import defaultdict
from utils.my_decorators import dividing_line
from utils.my_decorators import plt_support_cn
from utils.my_decorators import ignore_np_bool8_warning
from utils.my_path_utils import get_out_file_path


class BlackJackGame:
    def __init__(self, env=None, episode=1000, gamma=0.9, epsilon=0.1):
        self.trajectories = None
        self.win_or_loss = None
        self.win_rate = None
        self.state_value_table = None
        self.return_state_action = None
        self.q_table = None
        self.env = env
        self.episode = episode
        self.gamma = gamma
        self.epsilon = epsilon
        self._create_environment()

    def _create_environment(self):
        if not self.env:
            self.env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
        self.env.reset()

    def monte_carlo_simulation(self, with_offline_policy=False):
        if with_offline_policy:
            if not self.q_table:
                self.monte_carlo_simulation(with_offline_policy=False)

        trajectories = []
        win_or_loss = []
        for _ in tqdm(range(self.episode)):
            state, info = self.env.reset()
            trajectory = []

            while True:
                action = self._offline_policy(state) if with_offline_policy else self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

                if terminated or truncated:
                    win_or_loss.append(True if reward > 0 else False)
                    trajectories.append(trajectory)
                    break

        self.trajectories = trajectories
        self.win_or_loss = win_or_loss
        self.win_rate = sum(int(w) for w in self.win_or_loss) / len(self.win_or_loss)
        self._update_state_action_value()

    def _offline_policy(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(len(self.q_table[state]))
        else:
            return np.argmax(self.q_table[state])

    def _update_state_action_value(self):
        state_values = defaultdict(list)
        return_state_action = defaultdict(list)
        q_table = defaultdict(lambda: [0, 0])

        for t in self.trajectories:
            value = 0
            visited = set()
            for i, (s, a, r) in enumerate(t[::-1]):  # ????????????state???value????????????value????????????????????????gamma
                value = self.gamma * value + r
                if s not in visited:  # ???????????????????????????????????????state???
                    state_values[s].append(value)
                    return_state_action[(s, a)].append(value)
                    q_table[s][a] = np.mean(return_state_action[(s, a)])
                    visited.add(s)

        self.state_value_table = {s: np.mean(v) for s, v in
                                  sorted(state_values.items(), key=lambda item: np.mean(item[1]), reverse=True)}
        self.return_state_action = return_state_action
        self.q_table = q_table


class Task:
    @staticmethod
    @ignore_np_bool8_warning
    @dividing_line('??????????????????????????????')
    def monte_carlo_simulation_test(game=BlackJackGame()):
        game.monte_carlo_simulation()
        print('??????state?????????value:')
        for s, v in game.state_value_table.items():
            print('state: {}, value: {}'.format(s, v))

        print('???????????????: {}'.format(game.win_rate))
        max_state, max_value = max(game.state_value_table.items(), key=lambda x_y: x_y[1])
        print('???????????????state: {}, ??????value: {}'.format(max_state, max_value))

    @staticmethod
    @plt_support_cn
    @ignore_np_bool8_warning
    @dividing_line('????????????????????????????????????')
    def analyse_simulation_result(game=BlackJackGame()):

        def subplot(ax, state_table, ax_title):
            im = ax.imshow(state_table)
            plt.colorbar(im, ax=ax)
            ax.set_title(ax_title)
            ax.set(xlabel='????????????????????????', ylabel='????????????????????????')

        def subplot_3d(ax_3d, data_source, ax_title):
            xs, ys = np.meshgrid(np.arange(data_source.shape[0]),
                                 np.arange(data_source.shape[1]))
            zs = np.array([data_source[x][y] for x, y in zip(np.ravel(xs), np.ravel(ys))]).reshape(xs.shape)
            ax_3d.plot_wireframe(xs, ys, zs, rstride=1, cstride=1)
            ax_3d.set_title(ax_title)
            ax_3d.set(xlabel='????????????????????????', ylabel='????????????????????????', zlabel='state value')

        if not game.state_value_table:
            Task.monte_carlo_simulation_test(game=game)

        state_table_with_ace = np.zeros((30, 11))
        state_table_without_ace = np.zeros((30, 11))
        state_table_with_ace[:] = np.nan
        state_table_without_ace[:] = np.nan
        for (player, dealer_show, ace), value in game.state_value_table.items():
            if ace:
                state_table_with_ace[player][dealer_show] = value
            else:
                state_table_without_ace[player][dealer_show] = value

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('???????????????????????????????????????state?????????value', fontsize=15)
        subplot(ax1, state_table_with_ace, '???????????????ACE')
        subplot(ax2, state_table_without_ace, '??????????????????ACE')
        plt.tight_layout()
        fig_save_path = get_out_file_path('??????????????????????????????.png')
        plt.savefig(fig_save_path, dpi=600)
        plt.show()

        fig_3d, (ax1_3d, ax2_3d) = plt.subplots(1, 2, subplot_kw=dict(projection='3d'), figsize=(10, 5))
        fig_3d.suptitle('??????????????????????????????3D?????????state?????????value', fontsize=15)
        subplot_3d(ax1_3d, state_table_with_ace, '???????????????ACE')
        subplot_3d(ax2_3d, state_table_without_ace, '??????????????????ACE')
        plt.tight_layout()
        fig_3d_save_path = get_out_file_path('??????????????????????????????_3D.png')
        plt.savefig(fig_3d_save_path, dpi=600)
        plt.show()

    @staticmethod
    @ignore_np_bool8_warning
    @dividing_line('??????????????????????????????????????????')
    def monte_carlo_simulation_with_policy_test(game=BlackJackGame()):
        game.monte_carlo_simulation(with_offline_policy=True)
        print('???????????????: {}'.format(game.win_rate))


if __name__ == '__main__':
    ENV = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
    EPISODE = 500000
    BLACKJACK_GAME = BlackJackGame(env=ENV, episode=EPISODE)

    Task.monte_carlo_simulation_test(game=BLACKJACK_GAME)
    Task.analyse_simulation_result(game=BLACKJACK_GAME)
    Task.monte_carlo_simulation_with_policy_test(game=BLACKJACK_GAME)

    BLACKJACK_GAME.env.close()
