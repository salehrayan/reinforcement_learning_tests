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
env = gym.make("LunarLander-v2")
env.reset()

class DQN(nn.Module):

    def __init__(self, hidden_size, obs_size, n_actions, atoms=51):
        super().__init__()
        self.atoms = atoms
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n_actions * self.atoms)
        )

    def forward(self, x):
        x = self.net(x.float()).view(-1, self.n_actions, self.atoms)
        q_probs = F.softmax(x, dim=-1)
        return q_probs

def epsilon_greedy(state, net, n_actions, epsilon=0.0, support=0):
    if np.random.random() < epsilon:
        action = np.random.randint(n_actions)

    else:
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        q_value_probs = net(state)
        q_values = (support * q_value_probs).sum(dim=-1)
        _, action = torch.max(q_values, dim=-1)
        action = int(action.item())
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
    def __init__(self, max_steps, save_location, policy=epsilon_greedy, capacity=100_000,
                 batch_size=32, lr=1e-3, hidden_size=128, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=1,
                 eps_end=0.15, eps_last_episode=100, samples_per_epoch=10_000,
                 sync_rate=10, n_steps=3, v_min=-10, v_max=10, atoms=51):
        super().__init__()

        self.support = torch.linspace(v_min, v_max, atoms, device=device)
        self.delta = (v_max - v_min) / (atoms - 1)

        self.env = env
        self.n_actions = self.env.action_space.n

        obs_size = self.env.observation_space.shape[0]
        self.q_net = DQN(hidden_size, obs_size, self.n_actions, atoms=atoms)


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
                action = policy(state, self.q_net, n_actions=self.n_actions, epsilon=epsilon, support=self.support)
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
        self.log('epoch_reward_history', reward_accumulate)

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
        states, actions, returns, dones, next_states = batch
        returns = returns.unsqueeze(1)
        dones = dones.unsqueeze(1)
        batch_size = len(actions)

        q_value_probs = self.q_net(states)  # (B, A, N)
        action_value_probs = q_value_probs[range(batch_size), actions, :]
        log_action_value_probs = torch.log(action_value_probs + 1e-6)
        # state_action_values = self.q_net(states).gather(1, actions.type(torch.int64))

        with torch.no_grad():
            next_q_value_probs = self.target_q_net(next_states)
            next_q_values = (next_q_value_probs * self.support).sum(dim=-1)
            next_actions = next_q_values.argmax(dim=-1)

            next_action_values_probs = next_q_value_probs[range(batch_size), next_actions, :]

        m = torch.zeros(batch_size * self.hparams.atoms, device=device, dtype=torch.float64)

        Tz = returns + ~dones * self.hparams.gamma**self.hparams.n_steps * self.support.unsqueeze(0)
        Tz.clamp_(min=self.hparams.v_min, max=self.hparams.v_max)

        b = (Tz - self.hparams.v_min) / self.delta
        l, u = b.floor().long(), b.ceil().long()

        offset = torch.arange(batch_size, device=device).view(-1, 1) * self.hparams.atoms

        l_idx = (l + offset).flatten()
        u_idx = (u + offset).flatten()

        upper_probs = (next_action_values_probs * (u - b)).flatten()
        lower_probs = (next_action_values_probs * (b - l)).flatten()

        m.index_add_(dim=0, index=l_idx, source=upper_probs)
        m.index_add_(dim=0, index=u_idx, source=lower_probs)

        m = m.reshape(batch_size, self.hparams.atoms)

        cross_entropies = -(m* log_action_value_probs).sum(-1)
        loss = cross_entropies.mean()
        # self.log('episode/Q-Error', loss)
        return loss

    # Training epoch end
    def on_train_epoch_end(self):

        epsilon = max(self.hparams.eps_end,
                      self.hparams.eps_start -self.current_epoch / self.hparams.eps_last_episode)

        self.play_episode(policy=self.policy, epsilon=epsilon)

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        if self.episode_reward_history[-1] > self.episode_reward_history[-2] :
            torch.save({'q_net_state_dict': self.q_net.state_dict()}, self.hparams.save_location + 'q_net_ckpt.pth')

    def save_reward_plots(self, step_rewards, episode_rewards, location):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(episode_rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Episode Rewards Distributional_N_steps_DQN")
        plt.plot(episode_rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 20', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.tight_layout()
        plt.grid(True)

        plt.savefig(location + 'episode_rewards.png', format='png', dpi=1000, bbox_inches='tight')
        plt.close

        sma = np.convolve(step_rewards, np.ones(20) / 20, mode='valid')

        plt.figure()
        plt.title("Step Rewards Distributional_N_steps_DQN")
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


algorithm = DeepQLearning(1000, save_location='./Distributional_N_steps_DQN/')
early_stopping = EarlyStopping(monitor='epoch_reward_history', mode='max', patience=500)


trainer = Trainer(
    accelerator="cpu", max_epochs=10_000, callbacks=[early_stopping]
)

trainer.fit(algorithm)

env.close()