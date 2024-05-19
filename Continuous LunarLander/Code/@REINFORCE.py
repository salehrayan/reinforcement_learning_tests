import copy
import itertools
import random
import time
from collections import deque
import os

import gymnasium as gym
import torch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.distributions import Normal
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


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


env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=500)
env = RepeatActionWrapper(env, n=5)
env.reset()


device = 'cpu'



class GradientPolicy(nn.Module):
    def __init__(self, in_featrues, out_dim, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_featrues, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(hidden_size, out_dim)
        self.linear_std = nn.Linear(hidden_size, out_dim)

    def forward(self, x):
        x = self.net(torch.from_numpy(x).float().to(device))
        mu = torch.tanh(self.linear_mu(x))
        std = F.softplus(self.linear_std(x)) + 1e-3

        return mu, std



@torch.no_grad()
def test_agent(test_env, policy):
    obs = torch.from_numpy(test_env.reset()[0])
    rewards = 0
    done = False
    while not done:

        mu, std = policy(obs)
        action = torch.normal(mu, std)
        next_obs, reward, done1, done2, info = test_env.step(action.cpu().numpy())
        done = done1 | done2
        obs = next_obs
        obs = torch.from_numpy(obs).to(device)
        # action = torch.from_numpy(action).to(device)
        reward = torch.tensor(reward)
        done = torch.tensor(done)
        rewards += reward
    return rewards.mean()


class RLDataset(IterableDataset):

    def __init__(self, env, policy, episode_length, gamma):
        self.env = env
        self.policy = policy
        self.episode_length = episode_length
        self.gamma = gamma

        self.obs = torch.from_numpy(self.env.reset()[0]).to(device)

    @torch.no_grad()
    def __iter__(self):

        transitions = []

        for step in range(self.episode_length):

            mu, std = self.policy(self.obs)
            action = torch.normal(mu, std)
            next_obs, reward, done1, done2, info = self.env.step(action.cpu().numpy())
            done = done1 | done2
            self.obs = next_obs

            self.obs = torch.from_numpy(self.obs).to(device)
            # action = torch.from_numpy(action).to(device)
            reward = torch.tensor(reward)
            done = torch.tensor(done)
            transitions.append((self.obs, action, reward, done))


            obs_b, action_b, reward_b, done_b = map(torch.stack, zip(*transitions))

        running_return = torch.zeros((1), dtype=torch.float32, device=device)
        return_b = torch.zeros((self.episode_length, 1), dtype=torch.float32, device=device)
        for t, (obs, action, reward, done) in reversed(list(enumerate(transitions))):

            running_return = self.gamma * running_return * ~done + reward
            return_b[t, 0] = (self.gamma ** t) * running_return

        idx = list(range(self.episode_length))
        random.shuffle(idx)

        for i in idx:
            yield obs_b[i], action_b[i], return_b[i]


class Reinforce(LightningModule):

    def __init__(self, episode_length=500, batch_size=128, hidden_size=128, policy_lr=1e-4,
                 gamma=0.999, entropy_coef=1e-3, optim=AdamW):
        super().__init__()

        self.env = env
        test_env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=500)
        self.test_env = RepeatActionWrapper(test_env, n=7)


        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.dataset = RLDataset(self.env, self.policy, episode_length, gamma)

        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):

        obs_b, action_b, return_b = batch

        mu, std = self.policy(obs_b)
        dist = torch.distributions.Normal(mu, std)
        log_prob_b = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        policy_loss = -log_prob_b * return_b

        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        self.log('Policy loss', policy_loss.mean(), on_epoch=True)
        self.log('Entropy', entropy.mean(), on_epoch=True)

        return torch.mean(policy_loss - self.hparams.entropy_coef * entropy)

    def on_train_epoch_end(self):
        mean_rewards = test_agent(self.test_env, self.policy)
        self.log('mean_rewards', mean_rewards, on_epoch=True)


dir_path = "./REINFORCE_Lunar_results/"

early_stopping = EarlyStopping(monitor='mean_rewards', mode='max', patience=300)
tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
csv_logger = CSVLogger(dir_path, version='csv')
ckpt_callback = ModelCheckpoint(dir_path, monitor='mean_rewards', filename='{epoch}-{mean_rewards:.4f}', every_n_epochs =10, mode='max')


algorithm = Reinforce()

trainer = Trainer(accelerator='cpu',max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
    default_root_dir=dir_path, log_every_n_steps=1)

trainer.fit(algorithm)


env.close()