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
from pytorch_lightning.core.optimizer import LightningOptimizer
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


env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=300)
# env = RepeatActionWrapper(env, 5)
print(env.reset())

test_env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=300, render_mode="rgb_array")
# test_env = RepeatActionWrapper(test_env, 5)
test_env.reset()

device = 'cpu'
class GradientPolicy(nn.Module):
    def __init__(self, in_featrues, out_dims, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_featrues, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(hidden_size, out_dims)
        self.linear_std = nn.Linear(hidden_size, out_dims)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = self.net(torch.from_numpy(x).float().to(device))
        else:
            x = self.net(x.float().to(device))
        mu = torch.tanh(self.linear_mu(x))
        std = F.softplus(self.linear_std(x)) + 1e-3

        return mu, std


class ValueNet(nn.Module):

    def __init__(self, in_features, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_features, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        if isinstance(x, np.ndarray):
            return self.net(torch.from_numpy(x).float().to(device))
        else:
            return self.net(x.float().to(device))


class RLDataset(IterableDataset):

    def __init__(self, env, policy, value_net, samples_per_epoch, gamma, lamb, epoch_repeat):
        super().__init__()
        self.env = env
        self.policy = policy
        self.value_net = value_net
        self.gamma = gamma
        self.lamb = lamb
        self.samples_per_epoch = samples_per_epoch
        self.epoch_repeat = epoch_repeat
        self.obs = self.env.reset()[0]

    @torch.no_grad()
    def __iter__(self):
        self.obs = self.env.reset()[0]
        transitions = []
        for step in range(self.samples_per_epoch):
            mu, std = self.policy(self.obs)
            action = torch.normal(mu, std)
            next_obs, reward, done1, done2, info = self.env.step(action.cpu().numpy())
            done = done1 | done2
            transitions.append((self.obs, mu, std, action, reward, done, next_obs))
            self.obs = next_obs

        obs_b, mu_b, std_b, action_b, reward_b, done_b, next_obs_b  =  map(torch.from_numpy,  map(np.vstack, zip(*transitions)))


        values_b = self.value_net(obs_b)
        next_values_b = self.value_net(next_obs_b)

        td_error_b = reward_b + ~done_b * self.gamma * next_values_b - values_b

        running_gae = torch.zeros(1, dtype=torch.float32, device=device)
        gae_b = torch.zeros_like(td_error_b)

        for row in range(self.samples_per_epoch - 1, -1, -1):
            running_gae = td_error_b[row] + ~done_b[row] * self.gamma * self.lamb * running_gae
            gae_b[row] = running_gae

        target_b = gae_b + values_b

        for repeat in range(self.epoch_repeat):
            idx = list(range(self.samples_per_epoch))
            random.shuffle(idx)

            for i in idx:
                yield obs_b[i], mu_b[i], std_b[i], action_b[i], reward_b[i], gae_b[i], target_b[i]

@torch.no_grad()
def test_agent(env, policy, episodes=10):

  ep_returns = []
  for ep in range(episodes):
    state = env.reset()[0]
    done = False
    ep_ret = 0.0

    while not done:
      loc, scale = policy(state)
      sample = torch.normal(loc, scale)
      action = torch.tanh(sample)
      state, reward, done1, done2, info = env.step(action.numpy())
      done = done1 or done2
      ep_ret += reward

    ep_returns.append(ep_ret)

  return sum(ep_returns) / episodes

class PPO(LightningModule):

    def __init__(self, batch_size=100,  hidden_size=64, samples_per_epoch=300,
                 epoch_repeat=5, policy_lr=5e-5, value_lr=5e-5, gamma=0.99,
                 epsilon=0.3, entropy_coef=0.5, lamb=0.95, optim=AdamW):

        super().__init__()

        self.env = env
        self.test_env = test_env

        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.value_net = ValueNet(obs_size, hidden_size)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.dataset = RLDataset(self.env, self.policy, self.target_value_net, samples_per_epoch,
                                 gamma, lamb, epoch_repeat)
        self.save_hyperparameters()
        self.automatic_optimization = False

    def configure_optimizers(self):
        value_opt = LightningOptimizer(self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr))
        policy_opt = LightningOptimizer(self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr))
        return value_opt, policy_opt

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        value_opt, policy_opt = self.optimizers()

        obs_b, mu_b, std_b, action_b, reward_b, gae_b, target_b = batch

        state_values = self.value_net(obs_b)

        value_loss = F.smooth_l1_loss(state_values, target_b)
        self.log('episode/Value Loss', value_loss)
        value_opt.zero_grad()
        policy_opt.zero_grad()
        self.manual_backward(value_loss)
        value_opt.step()

        new_mu, new_std = self.policy(obs_b)
        dist = torch.distributions.Normal(new_mu, new_std)
        log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        prev_dist = Normal(mu_b, std_b)
        prev_log_prob = prev_dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        rho = torch.exp(log_prob - prev_log_prob)

        surrogate_1 = rho * gae_b
        surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * gae_b
        policy_loss = - torch.minimum(surrogate_1, surrogate_2)

        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        loss = policy_loss - self.hparams.entropy_coef * entropy

        self.log('episode/Policy Loss', policy_loss.mean())
        self.log('episode/Entropy', entropy.mean())
        self.log('episode/Reward', reward_b.float().mean())

        value_opt.zero_grad()
        policy_opt.zero_grad()
        self.manual_backward(loss.mean())
        policy_opt.step()

    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())

        if self.current_epoch % 1 == 0:
            average_return = test_agent(self.test_env, self.policy, episodes=1)
            self.log('Average_return', average_return)

dir_path = "C:/Users/ASUS/Desktop/Re-inforcement/some_gym_experiments/Continuous LunarLander/Code/PPO_GAE_Lunar_results"

early_stopping = EarlyStopping(monitor='Average_return', mode='max', patience=1000)
tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
csv_logger = CSVLogger(dir_path, version='csv')
ckpt_callback = ModelCheckpoint(dirpath=dir_path, monitor='Average_return', filename='{epoch}-{Average_return:.4f}',save_on_train_epoch_end=True, mode='max')


algorithm = PPO()

trainer = Trainer(accelerator='cpu',max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
    default_root_dir=dir_path)

trainer.fit(algorithm)


env.close()




