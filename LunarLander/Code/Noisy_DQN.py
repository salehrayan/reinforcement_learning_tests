import copy
import random
from collections import deque
import os

import gymnasium as gym
import torch

import numpy as np
import matplotlib.pyplot as plt
import math

import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_, zeros_
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(42)


device = 'cpu'
env = gym.make("LunarLander-v2")
env.reset()


class NoisyLinear(nn.Module):

    def __init__(self, in_features, out_features, sigma):
        super().__init__()
        self.w_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.w_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.b_mu = nn.Parameter(torch.empty(out_features))
        self.b_sigma = nn.Parameter(torch.empty(out_features))

        kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        kaiming_uniform_(self.w_sigma, a=math.sqrt(5))
        zeros_(self.b_mu)
        zeros_(self.b_sigma)

    def forward(self, x, sigma=0.5):
        if self.training:
            w_noise = torch.normal(0, sigma, size=self.w_mu.size()).to(device)
            b_noise = torch.normal(0, sigma, size=self.b_mu.size()).to(device)
            return F.linear(x, w_noise * self.w_sigma + self.w_mu, b_noise * self.b_sigma + self.b_mu)
        else:
            return F.linear(x, self.w_mu, self.b_mu)

class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, n_actions, sigma=0.5):
        super().__init__()
        self.net = nn.Sequential(
            NoisyLinear(obs_size, hidden_size, sigma),
            nn.ReLU(),
            NoisyLinear(hidden_size, hidden_size, sigma),
            nn.ReLU(),
            NoisyLinear(hidden_size, n_actions, sigma)
        )

    def forward(self, x):
        return self.net(x.float())

def greedy(state, net):
    state = torch.from_numpy(state).to(device).unsqueeze(0)
    q_values = net(state)
    _, action = torch.max(q_values, dim=1)
    action = int(action.item())  #realized that I'm in continious space not discrete
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



class DeepQLearning(LightningModule):

    # Initialize
    def __init__(self, max_steps, save_location, policy=greedy, capacity=100_000,
                 batch_size=32, lr=1e-3, hidden_size=128, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=1,
                 eps_end=0.15, eps_last_episode=100, samples_per_epoch=10_000,
                 sync_rate=10):
        super().__init__()
        self.env = env
        self.n_actions = self.env.action_space.n

        obs_size = self.env.observation_space.shape[0]
        self.q_net = DQN(hidden_size, obs_size, self.n_actions)


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
            self.play_episode()


    @torch.no_grad()
    def play_episode(self, policy=None):
        self.q_net.eval()
        state = self.env.reset()[0]
        done = False
        reward_accumulate = 0
        n_step = 0

        while not done and n_step < self.hparams.max_steps:
            if policy:
                action = policy(state, self.q_net)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done1, done2, info = self.env.step(action)
            # next_state = next_state
            done = done1 | done2
            exp = (state, action, reward, done, next_state)
            self.buffer.append(exp)
            state = next_state
            reward_accumulate = reward_accumulate + reward

            n_step += 1

            self.step_reward_history.append(reward)
        self.episode_reward_history.append(reward_accumulate)

        if (self.current_epoch+1) % 50 == 0:
            self.save_reward_plots(self.step_reward_history, self.episode_reward_history, self.hparams.save_location)


    # Forward
    def forward(self, x):
        return self.q_net(x)

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
        states, actions, rewards, dones, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        state_action_values = self.q_net(states).gather(1, actions.type(torch.int64))

        next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
        next_action_values[dones] = 0.0

        expected_state_action_values = rewards + self.hparams.gamma * next_action_values

        loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
        
        return loss

    # Training epoch end
    def on_train_epoch_end(self):

        self.play_episode(policy=self.policy)

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        if self.episode_reward_history[-1] > self.episode_reward_history[-2] :
            torch.save({'q_net_state_dict': self.q_net.state_dict()}, self.hparams.save_location + 'q_net_ckpt.pth')

        self.log('epoch_reward_history', self.episode_reward_history[-1])

    def save_reward_plots(self, step_rewards, episode_rewards, location):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(episode_rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Episode Rewards Noisy_DQN_results")
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
        plt.title("Step Rewards Noisy_DQN_results")
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


algorithm = DeepQLearning(1000, save_location='./Noisy_DQN_results/')
early_stopping = EarlyStopping(monitor='epoch_reward_history', mode='max', patience=500)


trainer = Trainer(
    accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping]
)

trainer.fit(algorithm)

env.close()