import random
from collections import deque
import os

import gymnasium as gym
from gymnasium.wrappers import NormalizeReward, NormalizeObservation, RecordVideo
import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt


class RepeatActionWrapper(gym.Wrapper):

    def __init__(self, env, n):
        super().__init__(env)
        self.env = env
        self.n = n

    def step(self, action):
        total_reward = 0.0

        for _ in range(self.n):
            next_state, reward, done1, done2, info = self.env.step(action)
            total_reward += reward
            if done1 | done2:
                break
        return next_state, total_reward, done1, done2, info

device = 'cpu'
env = gym.make("LunarLander-v2", continuous=True, render_mode='rgb_array')
env = RecordVideo(env, './video/', episode_trigger=lambda x: True)
env = RepeatActionWrapper(env, n=7)


class NafDQN(nn.Module):

    def __init__(self, hidden_size, obs_size, action_dims, max_action):
        super().__init__()
        self.action_dims = action_dims
        self.max_action = torch.from_numpy(max_action).to(device)
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.linear_mu = nn.Linear(hidden_size, action_dims)
        self.linear_value = nn.Linear(hidden_size, 1)
        self.linear_matrix = nn.Linear(hidden_size, int(action_dims * (action_dims + 1) / 2))

    # Mu: computes the action with the highest Q-value
    @torch.no_grad()
    def mu(self, x):
        x = self.net(x)
        x = self.linear_mu(x)
        x = torch.tanh(x) * self.max_action
        return x

    # Compute state value
    @torch.no_grad()
    def value(self, x):
        x = self.net(x)
        x = self.linear_value(x)
        return x

    def forward(self, x, a):
        x = self.net(x)
        mu = torch.tanh(self.linear_mu(x)) * self.max_action
        value = self.linear_value(x)

        # P(x)
        matrix = torch.tanh(self.linear_matrix(x))

        L = torch.zeros(x.shape[0], self.action_dims, self.action_dims).to(device)
        tril_indices = torch.tril_indices(row=self.action_dims, col=self.action_dims).to(device)
        L[:, tril_indices[0], tril_indices[1]] = matrix
        L.diagonal(dim1=1, dim2=2).exp_()
        P = L * L.transpose(2, 1)

        u_mu = (a-mu).unsqueeze(1)
        u_mu_t = u_mu.transpose(1, 2)

        adv = -1 / 2 * u_mu @ P @ u_mu_t
        adv = adv.squeeze(dim=-1)
        return value + adv


def noisy_policy(state, env, net, epsilon=0.0):

    state = torch.from_numpy(state).to(device).unsqueeze(0)
    amin = torch.from_numpy(env.action_space.low).to(device)
    amax = torch.from_numpy(env.action_space.high).to(device)
    mu = net.mu(state)
    mu = mu + torch.normal(mean=0, std=epsilon, size=mu.size()).to(device)
    action = mu.clamp(min=amin, max=amax)
    action = action.squeeze().cpu().numpy()
    return action


ckpt = torch.load(r'C:\Users\ASUS\Desktop\Re-inforcement\some_gym_experiments\Continuous LunarLander\Code\N_step_NAF_DQN_results\q_net_ckpt.pth')
q_net = NafDQN(512, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.high)
q_net.load_state_dict(ckpt['q_net_state_dict'])

for i in range(2):
    state = env.reset()[0]
    done = False

    while not done:
        action = noisy_policy(state, env, q_net)
        next_state, reward, done1, done2, info = env.step(action)

        done = done1 | done2

        state = next_state




