import copy
import random
from collections import deque
import os

import gymnasium as gym
import torch

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

"######################################################################"
env = gym.make("LunarLander-v2", continuous=True)
env = RepeatActionWrapper(env, n=7)
env.reset()
"######################################################################"

def polyak_averaging(net, target_network, tau=0.01):
    for qp, tp in zip(net.parameters(), target_network.parameters()):
        tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)

class NAFDeepQLearning(LightningModule):

    # Initialize
    def __init__(self, max_steps, save_location, policy=noisy_policy, capacity=100_000,
                 batch_size=32, lr=1e-4, hidden_size=512, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=2.0,
                 eps_end=0.2, eps_last_episode=1000, samples_per_epoch=1_000,
                 tau=0.01, n_steps=3):
        super().__init__()
        self.env = env
        self.action_dims = self.env.action_space.shape[0]
        self.max_actions = self.env.action_space.high

        obs_size = self.env.observation_space.shape[0]
        self.q_net = NafDQN(hidden_size, obs_size, self.action_dims, self.max_actions).to(device)


        self.target_q_net = copy.deepcopy(self.q_net)
        self.q_net.train()

        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        self.step_reward_history = []
        self.episode_reward_history = []
        os.makedirs(self.hparams.save_location, exist_ok=True)

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f"{len(self.buffer)} samples in experience buffer. Filling...")
            self.play_episode(epsilon=self.hparams.eps_start)


    @torch.no_grad()
    def play_episode(self, policy=None, epsilon=0.):
        self.q_net.eval()
        state = self.env.reset()[0]
        done = False
        reward_accumulate = 0
        n_step = 0
        transitions = []

        while not done and n_step < self.hparams.max_steps:
            if policy:
                action = policy(state, self.env, self.q_net, epsilon=epsilon)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done1, done2, info = self.env.step(action)
            # next_state = next_state
            done = done1 | done2
            exp = (state, action, reward, done, next_state)
            transitions.append(exp)
            state = next_state
            reward_accumulate = reward_accumulate + reward

            n_step += 1

            self.step_reward_history.append(reward)

        for i, (s, a, r, d, ns) in enumerate(transitions):
            batch = transitions[i:i + self.hparams.n_steps]
            ret = sum(t[2] * self.hparams.gamma**j for j,t in enumerate(batch))
            _, _, _, ld, ls = batch[-1]
            self.buffer.append((s, a, ret, ld, ls))

        self.episode_reward_history.append(reward_accumulate)
        # self.log('epoch_reward_history', reward_accumulate)

        if (self.current_epoch+1) % 50 == 0:
            self.save_reward_plots(self.step_reward_history, self.episode_reward_history, self.hparams.save_location)


    # Forward
    def forward(self, x):
        output = self.q_net.mu(x)
        return output

    # Configure optimizers
    def configure_optimizers(self):
        q_net_optimizer = self.hparams.optim(self.q_net.parameters(), lr=self.hparams.lr)
        return [q_net_optimizer]

    # Create dataloader
    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset = dataset,
            batch_size=self.hparams.batch_size
        )
        return dataloader

    # Training step
    def training_step(self, batch, batch_idx):
        self.q_net.train()
        states, actions, returns, dones, next_states = batch
        # actions = actions.unsqueeze(1)
        returns = returns.unsqueeze(1)
        dones = dones.unsqueeze(1)

        action_values = self.q_net(states, actions)

        next_state_values = self.target_q_net.value(next_states)
        next_state_values[dones] = 0.0

        target = returns + self.hparams.gamma * next_state_values

        loss = self.hparams.loss_fn(target, action_values)
        self.log('episode/Q-Error', loss)
        return loss

    # Training epoch end
    def on_train_epoch_end(self):

        epsilon = max(self.hparams.eps_end,
                      self.hparams.eps_start -self.current_epoch / self.hparams.eps_last_episode)

        self.play_episode(policy=self.policy, epsilon=epsilon)

        polyak_averaging(self.q_net, self.target_q_net, self.hparams.tau)
        if self.episode_reward_history[-1] > self.episode_reward_history[-2] :
            torch.save({'q_net_state_dict': self.q_net.state_dict()}, self.hparams.save_location + 'q_net_ckpt.pth')

        self.log('epoch_reward_history', self.episode_reward_history[-1])

    def save_reward_plots(self, step_rewards, episode_rewards, location):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(episode_rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Episode Rewards NAF_DQN")
        plt.plot(episode_rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 20', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(location + 'episode_rewards.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close()

        sma = np.convolve(step_rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Step Rewards NAF_DQN")
        plt.plot(step_rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 20', color='#f0c52b')
        plt.xlabel("Step")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(location + 'step_rewards.png', format='png', dpi=1000, bbox_inches='tight')


        # plt.show()
        plt.clf()
        plt.close()


algorithm = NAFDeepQLearning(1000, save_location='./NAF_DQN/')
early_stopping = EarlyStopping(monitor='epoch_reward_history', mode='max', patience=500)


trainer = Trainer(
    accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping]
)

trainer.fit(algorithm)

env.close()