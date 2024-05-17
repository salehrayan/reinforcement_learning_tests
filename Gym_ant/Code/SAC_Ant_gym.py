import copy
import itertools
import random
import time
from collections import deque
import os
import jax

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


env = gym.make('Ant-v4')
device='cpu'

class GradientPolicy(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims, max):
        super().__init__()
        self.max = torch.from_numpy(max).to(device)
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU())

        self.linear_mu = nn.Linear(hidden_size, out_dims)
        self.linear_std = nn.Linear(hidden_size, out_dims)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(device)
        x = self.net(obs.float())
        mu = self.linear_mu(x)
        std = self.linear_std(x)
        std = F.softplus(std) + 1e-3

        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=-1, keepdim=True)

        action = torch.tanh(action) * self.max

        return action, log_prob
class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size + out_dims, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(device)

        in_vector = torch.hstack((state, action))
        return self.net(in_vector.float())

class ReplayBuffer:

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RLDataset(IterableDataset):

    def __init__(self, buffer, sample_size=200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        for experience in self.buffer.sample(self.sample_size):
            yield experience


def polyak_averaging(net, target_network, tau=0.01):
    for qp, tp in zip(net.parameters(), target_network.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)


class SAC(LightningModule):

    def __init__(self, max_steps=1000, capacity=100_000, batch_size=128, lr=1e-3,
                 hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
                 epsilon=0.05, alpha=0.02, samples_per_epoch=128*10,
                 tau=0.01):
        super().__init__()
        self.env = env
        self.env.reset()
        # self.videos = []
        self.automatic_optimization = False

        obs_size = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        max_action = self.env.action_space.high

        self.q_net1 = DQN(hidden_size, obs_size, self.action_dims)
        self.q_net2 = DQN(hidden_size, obs_size, self.action_dims)
        self.policy = GradientPolicy(hidden_size, obs_size, self.action_dims, max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net1 = copy.deepcopy(self.q_net1)
        self.target_q_net2 = copy.deepcopy(self.q_net2)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f"{len(self.buffer)} samples in experience buffer. Filling...")
            self.play_episode()

    @torch.no_grad()
    def play_episode(self, policy=None):

        state = self.env.reset()[0]
        done = False
        reward_accumulate = 0
        n_step = 0

        while not done and n_step < self.hparams.max_steps:
            if policy and random.random() > self.hparams.epsilon:
                action, _ = policy(state)
                action = action.cpu().numpy()
            else:
                action = self.env.action_space.sample()
            next_state, reward, done1, done2, info = self.env.step(action)
            done = done1 | done2
            exp = (state, action, reward, done, next_state)
            self.buffer.append(exp)
            state = next_state
            reward_accumulate = reward_accumulate + reward

            n_step += 1
        return reward_accumulate.mean()
    def forward(self, x):
        output = self.policy.mu(x)
        return output

    def configure_optimizers(self):
        q_net_params = itertools.chain(self.q_net1.parameters(), self.q_net2.parameters())
        q_net_optimizers = self.hparams.optim(q_net_params, lr=self.hparams.lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.lr)
        return [q_net_optimizers, policy_optimizer]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size= self.hparams.batch_size
        )
        return dataloader

    def training_step(self, batch, batch_idx):

        q_net_optimizers, policy_optimizer = self.optimizers()

        states, actions, rewards, dones, next_states = map(torch.squeeze, batch)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).bool()


        action_values1 = self.q_net1(states, actions)
        action_values2 = self.q_net2(states, actions)
        target_actions, target_log_probs = self.target_policy(next_states)
        next_action_values = torch.min(self.target_q_net1(next_states, target_actions).detach(),
                                       self.target_q_net2(next_states, target_actions).detach())
        next_action_values[dones] = 0.
        target = rewards + self.hparams.gamma * (next_action_values - self.hparams.alpha * target_log_probs)

        q_loss1 = self.hparams.loss_fn(action_values1, target)
        q_loss2 = self.hparams.loss_fn(action_values2, target)
        q_loss = q_loss1 + q_loss2
        self.log('Q-loss', q_loss)
        q_net_optimizers.zero_grad()
        policy_optimizer.zero_grad()
        self.manual_backward(q_loss)
        q_net_optimizers.step()

        actions, log_probs = self.policy(states)
        action_values = torch.min(self.q_net1(states, actions),
                                  self.q_net2(states, actions))
        policy_loss = - (action_values - self.hparams.alpha * log_probs).mean()
        self.log('Policy-Loss', policy_loss)
        q_net_optimizers.zero_grad()
        policy_optimizer.zero_grad()
        self.manual_backward(policy_loss)
        policy_optimizer.step()


    def on_train_epoch_end(self):
        mean_rewards = self.play_episode(policy=self.policy)
        self.log("mean_rewards", mean_rewards, on_epoch=True)

        polyak_averaging(self.q_net1, self.target_q_net2, tau=self.hparams.tau)
        polyak_averaging(self.q_net2, self.target_q_net2, tau=self.hparams.tau)
        polyak_averaging(self.policy, self.target_policy, tau=self.hparams.tau)


dir_path = "./SAC_Ant_results/"

algorithm = SAC(alpha=0.005, tau=0.1)

early_stopping = EarlyStopping(monitor='mean_rewards', mode='max', patience=500)
tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
csv_logger = CSVLogger(dir_path, version='csv')
ckpt_callback = ModelCheckpoint(dir_path, monitor='mean_rewards', filename='{epoch}-{mean_rewards:.4f}', every_n_epochs =10, mode='max')

# algorithm.load_from_checkpoint()
trainer = Trainer(
    accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
    default_root_dir=dir_path, log_every_n_steps=1
)


trainer.fit(algorithm)

env.close()