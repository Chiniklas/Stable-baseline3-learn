from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from shower import ShowerEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os


env = ShowerEnv()
# print(env.observation_space.sample())
# print(env.action_space.sample())

# random steps tests
episodes = 10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state,reward,done,info = env.step(action)
        score+=reward
    print('episode:{} score:{}'.format(episode,score))

tensorboard_log = os.path.join("ppo_training",'best_model')
model = PPO("MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            learning_rate=0.0003)
model.learn(total_timesteps=int(1e5))
