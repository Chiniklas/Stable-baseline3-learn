import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import torch

# environment_name = 'CartPole-v0'
# env = gym.make(environment_name)
#
# print(environment_name)
#
# print(env.action_space.sample())
#
# print(env.action_space)
# print(env.observation_space)

print(torch.__version__)
print(torch.cuda.is_available())
