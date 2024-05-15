import copy
import functools
import random
from collections import deque
import os
import jax

import gymnasium as gym
import torch
import brax
from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import image

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping


device = 'cpu'

env = envs.create('ant',batch_size=512, episode_length=1000)
env = gym_wrapper.VectorGymWrapper(env)
env = torch_wrapper.TorchWrapper(env, device=device)


@torch.no_grad()
def test_env(env_name, policy=None):
  env = envs.create('ant', batch_size=1, episode_length=1000)
  env = gym_wrapper.VectorGymWrapper(env)
  env = torch_wrapper.TorchWrapper(env, device=device)
  qp_array = []
  state = env.reset()
  for i in range(1000):
    if policy:
      action = policy.mu(state)
    else:
      action = env.action_space.sample()
    state, _, _, _ = env.step(action)
    qp_array.append(env.render('rgb_array'))
  return qp_array
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

    def __init__(self, save_location, capacity=10_000, batch_size=512, actor_lr=1e-4, critic_lr=1e-4,
                 hidden_size=256, gamma=0.99, loss_fn=F.smooth_l1_loss, optim=AdamW,
                 eps_start=1, eps_end=0.2, eps_last_episode=500, samples_per_epoch=20,
                 tau=0.01):
        super().__init__()
        self.env = env
        self.obs = self.env.reset()
        self.videos = []
        self.automatic_optimization = False

        obs_size = self.env.observation_space.shape[1]
        self.action_dims = self.env.action_space.shape[1]
        min_action = self.env.action_space.low
        max_action = self.env.action_space.high

        self.q_net = DQN(hidden_size, obs_size, self.action_dims)
        self.policy = GradientPolicy(hidden_size, obs_size, self.action_dims, min_action, max_action)

        self.target_policy = copy.deepcopy(self.policy)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        self.reward_history = []
        os.makedirs(self.hparams.save_location, exist_ok=True)

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f"{len(self.buffer)} samples in experience buffer. Filling...")
            self.play(epsilon=self.hparams.eps_start)

    @torch.no_grad()
    def play(self, policy=None, epsilon=0.):
        self.q_net.eval()
        if policy:
            action = policy(self.obs, epsilon=epsilon)
        else:
            action = env.action_space.sample()
        next_obs, reward, done, info = self.env.step(torch.from_numpy(action))

        exp = (self.obs, action, reward, done, next_obs)
        self.buffer.append(exp)
        self.obs = next_obs
        reward_mean =  reward.mean()

        self.reward_history.append(reward_mean)

        if (self.current_epoch+1) % 50 == 0:
            self.save_reward_plots(self.reward_history, self.hparams.save_location)
        return reward_mean

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
            batch_size=1
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
        next_action_values = self.target_q_net(next_states, next_actions)
        next_action_values[dones] = 0.

        target = rewards + self.hparams.gamma * next_action_values
        q_loss = self.hparams.loss_fn(action_values, target)
        self.log('Q-loss', q_loss)
        q_loss.backward()
        q_net_optimizers.step()
        q_net_optimizers.zero_grad()



        mu = self.policy.mu(states)
        policy_loss = -self.q_net(states, mu).mean()
        self.log('Policy-Loss', policy_loss)
        policy_loss.backward()
        policy_optimizer.step()
        policy_optimizer.zero_grad()


    def on_train_epoch_end(self):
        # if self.current_epoch % 100 == 0:
        #     video = test_env('ant', self.policy)
        #     self.videos.append(video)

        if self.reward_history[-1] > self.reward_history[-2] :
            torch.save({'q_net_state_dict': self.q_net.state_dict()}, self.hparams.save_location + 'q_net_ckpt.pth')

    def save_reward_plots(self, rewards, location):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Episode Rewards DDPG_ant_brax")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 20', color='#f0c52b')
        plt.xlabel("Steps")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(location + 'Step_rewards.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close()


algorithm = DDPG(save_location='./DDPG_ant_brax/')
early_stopping = EarlyStopping(monitor='mean_rewards', mode='max', patience=500)


trainer = Trainer(
    accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping]
)

trainer.fit(algorithm)

env.close()