import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make("LunarLander-v2", continuous=True, max_episode_steps=500, render_mode="rgb_array")
env.reset()

dir_path = './REINFORCE_Lunar_results/'
new_logger = configure(dir_path, ["csv", "tensorboard"])

# model = A2C("MlpPolicy", env, verbose)
# model.set_logger(new_logger)
#
# model.learn(total_timesteps=10_000, progress_bar=True)
# model.save("A2C_LunarLander")

model = A2C.load("A2C_LunarLander", env=env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)


print(mean_reward, std_reward)

