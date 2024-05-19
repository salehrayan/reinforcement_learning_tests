import copy
import itertools
import random
import time
from collections import deque
import os

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation
import torch

import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.distributions import Normal
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW, Optimizer

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
env = NormalizeObservation(env)
# env = RepeatActionWrapper(env, 5)
# print(env.reset())

test_env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=300, render_mode="rgb_array")
test_env = NormalizeObservation(test_env)
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


# Adapted from: https://github.com/rlworkgroup/garage/blob/master/src/garage/torch/optimizers/conjugate_gradient_optimizer.py

def unflatten_tensors(flattened, tensor_shapes):
    flattened = flattened.cpu()
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [
        np.reshape(pair[0], pair[1]).to(device)
        for pair in zip(np.split(flattened, indices), tensor_shapes)
    ]


def _build_hessian_vector_product(func, params, reg_coeff=1e-5):
    param_shapes = [p.shape or torch.Size([1]) for p in params]
    f = func()
    f_grads = torch.autograd.grad(f, params, create_graph=True)

    def _eval(vector):
        unflatten_vector = unflatten_tensors(vector, param_shapes)

        assert len(f_grads) == len(unflatten_vector)
        grad_vector_product = torch.sum(
            torch.stack(
                [torch.sum(g * x) for g, x in zip(f_grads, unflatten_vector)]))

        hvp = list(
            torch.autograd.grad(grad_vector_product, params,
                                retain_graph=True))
        for i, (hx, p) in enumerate(zip(hvp, params)):
            if hx is None:
                hvp[i] = torch.zeros_like(p)

        flat_output = torch.cat([h.reshape(-1) for h in hvp])
        return flat_output + reg_coeff * vector

    return _eval


def _conjugate_gradient(f_Ax, b, cg_iters, residual_tol=1e-10):
    p = b.clone()
    r = b.clone()
    x = torch.zeros_like(b)
    rdotr = torch.dot(r, r)

    for _ in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


class ConjugateGradientOptimizer(Optimizer):

    def __init__(self, params, max_constraint_value, cg_iters=10, max_backtracks=15,
                 backtrack_ratio=0.8, hvp_reg_coeff=1e-5, accept_violation=False):

        super().__init__(params, {})
        self._max_constraint_value = max_constraint_value
        self._cg_iters = cg_iters
        self._max_backtracks = max_backtracks
        self._backtrack_ratio = backtrack_ratio
        self._hvp_reg_coeff = hvp_reg_coeff
        self._accept_violation = accept_violation

    def step(self, closure):
        f_loss, f_constraint = closure()
        params = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    grads.append(p.grad.reshape(-1))

        flat_loss_grads = torch.cat(grads)
        f_Ax = _build_hessian_vector_product(f_constraint, params, self._hvp_reg_coeff)
        step_dir = _conjugate_gradient(f_Ax, flat_loss_grads, self._cg_iters)

        step_dir[step_dir.ne(step_dir)] = 0.

        step_size = np.sqrt(
            2.0 * self._max_constraint_value * (1. / (torch.dot(step_dir, f_Ax(step_dir)) + 1e-8)).cpu())

        if np.isnan(step_size):
            step_size = 1.

        descent_step = step_size * step_dir
        self._backtracking_line_search(params, descent_step, f_loss, f_constraint)

    def _backtracking_line_search(self, params, descent_step, f_loss, f_constraint):
        prev_params = [p.clone() for p in params]
        ratio_list = self._backtrack_ratio ** np.arange(self._max_backtracks)
        loss_before = f_loss()

        param_shapes = [p.shape or torch.Size([1]) for p in params]
        descent_step = unflatten_tensors(descent_step, param_shapes)
        assert len(descent_step) == len(params)

        for ratio in ratio_list:
            for step, prev_param, param in zip(descent_step, prev_params, params):
                step = ratio * step
                new_param = prev_param.data - step
                param.data = new_param.data

            loss = f_loss()
            constraint_val = f_constraint()
            if (loss < loss_before and constraint_val <= self._max_constraint_value):
                break




class RLDataset(IterableDataset):

    def __init__(self, env, policy, value_net, samples_per_epoch, gamma, lamb):
        super().__init__()
        self.env = env
        self.policy = policy
        self.value_net = value_net
        self.gamma = gamma
        self.lamb = lamb
        self.samples_per_epoch = samples_per_epoch
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
            if done:
                next_obs = self.env.reset()[0]
                transitions.append((self.obs, mu, std, action, reward, done, next_obs))
            else:
                self.obs = next_obs
                transitions.append((self.obs, mu, std, action, reward, done, next_obs))

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


class TRPO(LightningModule):

    def __init__(self, batch_size=2048, hidden_size=256, samples_per_epoch=15_000,
                  value_lr=1e-3, gamma=0.99,
                 epsilon=0.3, lamb=0.95, kl_limit=0.10, v_optim=AdamW, pi_optim=ConjugateGradientOptimizer):

        super().__init__()

        self.env = env
        self.test_env = test_env

        obs_size = self.env.observation_space.shape[0]
        action_dims = self.env.action_space.shape[0]

        self.policy = GradientPolicy(obs_size, action_dims, hidden_size)
        self.value_net = ValueNet(obs_size, hidden_size)
        self.target_value_net = copy.deepcopy(self.value_net)

        self.dataset = RLDataset(self.env, self.policy, self.target_value_net, samples_per_epoch,
                                 gamma, lamb)
        self.save_hyperparameters()
        self.automatic_optimization = False

    def configure_optimizers(self):
        value_opt = LightningOptimizer(self.hparams.v_optim(self.value_net.parameters(), lr=self.hparams.value_lr))
        policy_opt = LightningOptimizer(self.hparams.pi_optim(self.policy.parameters(), max_constraint_value=self.hparams.kl_limit))
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
        self.manual_backward(value_loss)
        value_opt.step()

        new_mu, new_std = self.policy(obs_b)
        dist = torch.distributions.Normal(new_mu, new_std)
        log_prob = dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        prev_dist = Normal(mu_b, std_b)
        prev_log_prob = prev_dist.log_prob(action_b).sum(dim=-1, keepdim=True)

        def loss_fn():
            loss = -torch.exp(log_prob - prev_log_prob) * gae_b
            return loss.mean()

        def constraint_fn():
            constraint = torch.distributions.kl_divergence(prev_dist, dist).sum(dim=-1)
            return constraint.mean()

        closure = lambda: (loss_fn, constraint_fn)

        loss = loss_fn()

        policy_opt.zero_grad()
        self.manual_backward(loss, retain_graph=True)
        policy_opt.step(closure)

        self.log('episode/Policy Loss', loss)
        self.log('episode/Reward', reward_b.float().mean())

    def on_train_epoch_end(self):
        if self.current_epoch % 1 == 0:
            self.target_value_net.load_state_dict(self.value_net.state_dict())

        if self.current_epoch % 1 == 0:
            average_return = test_agent(self.test_env, self.policy, episodes=1)
            self.log('Average_return', average_return)


dir_path = "C:/Users/ASUS/Desktop/Re-inforcement/some_gym_experiments/Continuous LunarLander/Code/TRPO_GAE_Lunar_results"

early_stopping = EarlyStopping(monitor='Average_return', mode='max', patience=1000)
tb_logger = TensorBoardLogger(dir_path, version='tensorboard')
csv_logger = CSVLogger(dir_path, version='csv')
ckpt_callback = ModelCheckpoint(dirpath=dir_path, monitor='Average_return', filename='{epoch}-{Average_return:.4f}',save_on_train_epoch_end=True, mode='max')


algorithm = TRPO()

trainer = Trainer(accelerator='cpu',max_epochs=10_000, callbacks=[early_stopping, ckpt_callback], logger=[tb_logger, csv_logger],
    default_root_dir=dir_path)

trainer.fit(algorithm)


env.close()







