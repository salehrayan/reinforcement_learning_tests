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


env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=500, render_mode="rgb_array")
# env = RepeatActionWrapper(env, 5)
env.reset()

test_env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=500, render_mode="rgb_array")
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

    def __init__(self, env, policy, samples_per_epoch, epoch_repeat):
        super().__init__()
        self.env = env
        self.policy = policy
        self.samples_per_epoch = samples_per_epoch
        self.epoch_repeat = epoch_repeat
        self.obs = self.env.reset()[0]

    @torch.no_grad()
    def __iter__(self):
        transitions =[]
        for step in range(self.samples_per_epoch):
            mu, std = self.policy(self.obs)
            action = torch.normal(mu, std).numpy()
            next_obs, reward, done1, done2, info = self.env.step(action)
            done = done1 or done2
            # if done:
            #     next_obs = self.env.reset()[0]
            # else:
            self.obs = next_obs
            transitions.append((self.obs, mu, std, action, reward, done, next_obs))



        obs_b, mu_b, std_b, action_b, reward_b, done_b, next_obs_b = map(torch.from_numpy,
                                                                    map(np.vstack, zip(*transitions)))

        for repeat in range(self.epoch_repeat):
            idx = list(range(len(reward_b)))
            random.shuffle(idx)

            for i in idx:
                yield obs_b[i], mu_b[i], std_b[i], action_b[i], reward_b[i], done_b[i], next_obs_b[i]

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

    def __init__(self, batch_size=64,  hidden_size=64, samples_per_epoch=500,
                 epoch_repeat=5, policy_lr=1e-3, value_lr=1e-3, gamma=0.99,
                 epsilon=0.3, entropy_coef=0.01, optim=AdamW):

        super().__init__()

        self.env = env
        self.test_env = test_env

        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.value_net = ValueNet(obs_size, hidden_size)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.dataset = RLDataset(self.env, self.policy, samples_per_epoch, epoch_repeat)
        self.save_hyperparameters()
        self.automatic_optimization = False

    def configure_optimizers(self):
        value_opt = self.hparams.optim(self.value_net.parameters(), lr=self.hparams.value_lr)
        policy_opt = self.hparams.optim(self.policy.parameters(), lr=self.hparams.policy_lr)
        return value_opt, policy_opt


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        value_opt, policy_opt = self.configure_optimizers()

        obs_b, mu_b, std_b, action_b, reward_b, done_b, next_obs_b = batch

        state_values = self.value_net(obs_b)

        with torch.no_grad():
            next_state_values = self.target_value_net(next_obs_b)
            next_state_values[done_b] = 0.
            target = reward_b + self.hparams.gamma * next_state_values

        value_loss = F.smooth_l1_loss(state_values, target)
        self.log('episode/Value_loss', value_loss)
        value_opt.zero_grad()
        policy_opt.zero_grad()
        value_loss.backward()
        value_opt.step()


        advantages = (target - state_values).detach()

        new_mu, new_std = self.policy(obs_b)
        dist = torch.distributions.Normal(new_mu, new_std)
        log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        prev_dist = torch.distributions.Normal(mu_b, std_b)
        prev_log_prob = prev_dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        rho = torch.exp(log_prob - prev_log_prob)

        surrogate_1 = rho * advantages
        surrogate_2 = rho.clip(1 - self.hparams.epsilon, 1 + self.hparams.epsilon) * advantages

        policy_loss = - torch.minimum(surrogate_1, surrogate_2)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        loss = (policy_loss - self.hparams.entropy_coef * entropy).mean()

        self.log('episode/Policy_loss', policy_loss.mean())
        self.log('episode/Entropy', entropy.mean())
        self.log('episode/Reward', reward_b.float().mean())

        value_opt.zero_grad()
        policy_opt.zero_grad()
        loss.backward()
        policy_opt.step()

    def on_train_epoch_end(self):
        self.target_value_net.load_state_dict(self.value_net.state_dict())

        if self.current_epoch % 10 == 0:
            average_return = test_agent(self.test_env, self.policy, episodes=1)
            self.log('episode/Average_return', average_return)



dir_path = "./PPO_Lunar_results/"

early_stopping = EarlyStopping(monitor='episode/Reward', mode='max', patience=300)
tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
csv_logger = CSVLogger(dir_path, version='csv')
ckpt_callback = ModelCheckpoint(dir_path, monitor='episode/Reward', filename='{epoch}-{episode/Reward:.4f}', every_n_epochs =10, mode='max')


algorithm = PPO()

trainer = Trainer(accelerator='cpu',max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
    default_root_dir=dir_path)

trainer.fit(algorithm)


env.close()




