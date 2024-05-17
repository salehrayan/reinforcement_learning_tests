import copy
import functools
import random
import time
from collections import deque
import os
import jax

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger


env = gym.make('Ant-v4', render_mode='rgb_array')
# env = RecordVideo(env,video_folder='./DDPG_Ant_results/', episode_trigger= lambda x:True, name_prefix='DDPG_Ant_randomAction')
device='cpu'

class GradientPolicy(nn.Module):

    def __init__(self, hidden_size, obs_size, out_dims, min, max):
        super().__init__()
        self.min = torch.from_numpy(min).to(device)
        self.max = torch.from_numpy(max).to(device)
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, out_dims),
                                 nn.Tanh())

    def mu(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(device)
        return self.net(x.float()) * self.max[0]

    def forward(self, x, epsilon):
        mu = self.mu(x)
        mu = mu + torch.normal(0, epsilon, size=mu.size(), device=mu.device)
        action = mu.clamp(min=self.min, max=self.max)
        action = action.cpu().numpy()
        return action

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


class DDPG(LightningModule):

    def __init__(self, max_steps=1000, capacity=1_000_000, batch_size=64, actor_lr=1e-3, critic_lr=1e-3,
                 hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
                 eps_start=1, eps_end=0.2, eps_last_episode=500, samples_per_epoch=64*100,
                 tau=0.01):
        super().__init__()
        self.env = env
        self.env.reset()
        # self.videos = []
        self.automatic_optimization = False

        obs_size = self.env.observation_space.shape[0]
        self.action_dims = self.env.action_space.shape[0]
        min_action = self.env.action_space.low
        max_action = self.env.action_space.high

        self.q_net = DQN(hidden_size, obs_size, self.action_dims)
        self.policy = GradientPolicy(hidden_size, obs_size, self.action_dims, min_action, max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        # while len(self.buffer) < self.hparams.samples_per_epoch:
        #     print(f"{len(self.buffer)} samples in experience buffer. Filling...")
        #     self.play(epsilon=self.hparams.eps_start)

    @torch.no_grad()
    def play(self, policy=None, epsilon=0.):

        state = self.env.reset()[0]
        done = False
        reward_accumulate = 0
        n_step = 0

        while not done and n_step < self.hparams.max_steps:
            if policy:
                action = policy(state, epsilon=epsilon)
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
        q_net_optimizers = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.critic_lr)
        policy_optimizer = self.hparams.optim(self.policy.parameters(), lr=self.hparams.actor_lr)
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

        epsilon = max(self.hparams.eps_end,
                      self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode)

        mean_rewards = self.play(policy=self.policy, epsilon=epsilon)
        self.log("mean_rewards", mean_rewards)

        polyak_averaging(self.q_net, self.target_q_net, tau=self.hparams.tau)
        polyak_averaging(self.policy, self.target_policy, tau=self.hparams.tau)

        states, actions, rewards, dones, next_states = map(torch.squeeze, batch)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).bool()


        action_values = self.q_net(states, actions)
        next_actions = self.target_policy.mu(next_states)
        next_action_values = self.target_q_net(next_states, next_actions).detach()
        next_action_values[dones] = 0.

        target = rewards + self.hparams.gamma * next_action_values
        q_loss = self.hparams.loss_fn(action_values, target)
        self.log('Q-loss', q_loss)
        q_net_optimizers.zero_grad()
        policy_optimizer.zero_grad()
        self.manual_backward(q_loss)
        q_net_optimizers.step()


        mu = self.policy.mu(states)
        policy_loss = -self.q_net(states, mu).mean()
        self.log('Policy-Loss', policy_loss)
        q_net_optimizers.zero_grad()
        policy_optimizer.zero_grad()
        self.manual_backward(policy_loss)
        policy_optimizer.step()



dir_path = "./DDPG_Ant_results/"

algorithm = DDPG.load_from_checkpoint(r'C:\Users\ASUS\Downloads\epoch=399-mean_rewards=661.2233.ckpt')

state = env.reset()[0]
with torch.no_grad():
    for _ in range(1000):
        action = algorithm.policy.mu(state).numpy()
        next_state, _, done1, done2, _ = env.step(env.action_space.sample())

        if done1 | done2:
            state = env.reset()[0]
            print('done')
        else:
            state = next_state
        # time.sleep(0.01)
# early_stopping = EarlyStopping(monitor='mean_rewards', mode='max', patience=500)
# tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
# csv_logger = CSVLogger(dir_path, version='csv')
# ckpt_callback = ModelCheckpoint(dir_path, monitor='mean_rewards', filename='{epoch}-{mean_rewards:.4f}', every_n_epochs =50, mode='max')
#
# # algorithm.load_from_checkpoint()
# trainer = Trainer(
#     accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
#     default_root_dir=dir_path, log_every_n_steps=1
# )
#
#
# trainer.fit(algorithm)

env.close()