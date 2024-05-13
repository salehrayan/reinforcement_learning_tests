import random
from collections import deque
import os

import gymnasium as gym
from gymnasium.wrappers import NormalizeReward, NormalizeObservation, RecordVideo
import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt


device = 'cpu'
env = gym.make("LunarLander-v2", render_mode='rgb_array')
env = RecordVideo(env, video_folder='./DoubleDQN_video/', episode_trigger= lambda x: True)

class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


def epsilon_greedy(state, net, n_actions, epsilon=0.0):
    if np.random.random() < epsilon:
        action = np.random.randint(n_actions)

    else:
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())  #realized that I'm in continious space not discrete
    return action

checkpoint = torch.load('./DoubleDQN_results/DoubleDQN_resultsq_net_ckpt.pth')
q_net = DQN(128, env.observation_space.shape[0], env.action_space.n)
q_net.load_state_dict(checkpoint['q_net_state_dict'])

for i in range(1):
    state = env.reset()[0]
    done = False
    frames = []

    while not done:
        action = epsilon_greedy(state, q_net, env.action_space.n)
        next_state, reward, done1, done2, info = env.step(action)

        done = done1 | done2

        state = next_state




