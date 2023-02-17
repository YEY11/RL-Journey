# -*- coding: utf-8 -*- 
# @Date : 2023/2/12
# @Author : YEY
# @File : blackjack_offline.py

import gymnasium as gym
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
from collections import defaultdict
from utils.my_decorators import dividing_line


class BlackJackGame:
    def __init__(self, env_natural=False, env_sab=False, env_render_mode='rgb_array'):
        self.trajectories = None
        self.win_or_loss = None
        self.win_rate = None
        self.state_value_table = None
        self.env = gym.make('Blackjack-v1', natural=env_natural, sab=env_sab, render_mode=env_render_mode)
        self.env.reset()

    def monte_carlo_simulation(self, episode=1000):
        trajectories = []
        win_or_loss = []
        for _ in tqdm(range(episode)):
            state, info = self.env.reset()
            trajectory = []

            while True:
                action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state

                if terminated or truncated:
                    win_or_loss.append(True if reward > 0 else False)
                    trajectories.append(trajectory)
                    break

        self.trajectories = trajectories
        self.win_or_loss = win_or_loss
        self.env.close()

    def get_win_rate(self):
        if not self.win_or_loss:
            self.monte_carlo_simulation()
        self.win_rate = sum(int(w) for w in self.win_or_loss) / len(self.win_or_loss)
        return self.win_rate

    def get_state_value_table(self, gamma=0.9):
        if not self.trajectories:
            self.monte_carlo_simulation()

        state_values = defaultdict(list)
        for t in self.trajectories:
            value = 0  # goal  / value
            visited = set()
            for i, (s, a, r) in enumerate(t[::-1]):  # why do this?
                value = gamma * value + r
                if s not in visited:
                    state_values[s].append(value)
                    visited.add(s)

        self.state_value_table = {s: np.mean(v) for s, v in
                                  sorted(state_values.items(), key=lambda item: np.mean(item[1]), reverse=True)}
        return self.state_value_table


class Task:
    @staticmethod
    @dividing_line('蒙特卡洛模拟测试')
    def monte_carlo_simulation_test():
        game = BlackJackGame()
        game.monte_carlo_simulation(episode=500000)
        state_value_table = game.get_state_value_table()
        print('不同state对应的value：')
        for s, v in state_value_table.items():
            print('state: {}, value: {}'.format(s, v))

        print('='*50)
        print(max(state_value_table.items(), key=lambda x_y: x_y[1]))





if __name__ == '__main__':
    Task.monte_carlo_simulation_test()
