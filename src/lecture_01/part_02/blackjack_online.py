# -*- coding: utf-8 -*- 
# @Date : 2023/2/18
# @Author : YEY
# @File : blackjack_online.py

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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
        self.policy = None
        self.env = env
        self.episode = episode
        self.gamma = gamma
        self.epsilon = epsilon
        self._create_environment()

    def _create_environment(self):
        if not self.env:
            self.env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
        self.env.reset()

    def monte_carlo_simulation_with_online_policy(self):
        self.return_state_action = defaultdict(list)
        self.q_table = defaultdict(lambda: [0, 0.])
        self.policy = defaultdict(lambda: [0, 0.])

        trajectories = []
        win_or_loss = []
        for _ in tqdm(range(self.episode)):
            state, info = self.env.reset()
            trajectory = []
            state_action_set = set()

            # 根据初始policy生成一个trajectory的(s,a,r)序列
            while True:
                action = np.argmax(self.policy[state])
                state_action_set.add((state, action))
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state
                if terminated or truncated:
                    win_or_loss.append(True if reward > 0 else False)
                    trajectories.append(trajectory)
                    break

            # 每轮episode都会更新policy
            value = 0
            for i, (s, a, r) in enumerate(trajectory[::-1]):
                value = self.gamma * value + r
                if (s, a) not in state_action_set:
                    self.return_state_action[(s, a)].append(value)
                    self.q_table[s][a] = float(np.mean(self.return_state_action[(s, a)]))
                    a_star = np.argmax(self.q_table[s])
                    action_set = set(range(len(self.policy[s])))
                    for act in action_set:
                        if act == a_star:
                            self.policy[s][act] = 1 - self.epsilon + self.epsilon / len(action_set)
                        else:
                            self.policy[s][act] = self.epsilon / len(action_set)

        self.trajectories = trajectories
        self.win_or_loss = win_or_loss
        self.win_rate = sum(int(w) for w in self.win_or_loss) / len(self.win_or_loss)


class Task:
    @staticmethod
    @ignore_np_bool8_warning
    @dividing_line('蒙特卡洛基于在线策略模拟试验')
    def monte_carlo_simulation_with_online_policy_test():
        env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
        episode = 500000
        game = BlackJackGame(env=env, episode=episode)
        game.monte_carlo_simulation_with_online_policy()
        print('玩家胜率为: {}'.format(game.win_rate))
        game.env.close()

    @staticmethod
    @ignore_np_bool8_warning
    @plt_support_cn
    @dividing_line('不同episode下的平均胜率分析')
    def different_episode_analysis_test():
        episodes = np.arange(5000, 505000, 5000)
        win_rates = []
        for e in episodes:
            env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode='rgb_array')
            game = BlackJackGame(env=env, episode=e)
            game.monte_carlo_simulation_with_online_policy()
            win_rates.append(game.win_rate)
            game.env.close()

        fig_name = '玩家胜率随着episode的变化趋势'
        plt.figure(figsize=(10, 6))
        plt.title(fig_name)
        plt.plot(episodes, win_rates, 'o-')
        out_file_path = get_out_file_path(fig_name + '.png')
        plt.savefig(out_file_path, dpi=600)
        plt.show()


if __name__ == '__main__':
    Task.monte_carlo_simulation_with_online_policy_test()
    Task.different_episode_analysis_test()
