import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import project3_RL_custum_env

# env = make_atari_env('Breakout-v0',n_envs=1,seed=0)
# env = VecFrameStack(env,n_stack=4)
#
# a2c_path = os.path.join('Training','Saved Models','A2C_Breakout_Model')
# model = A2C.load(a2c_path,env)
#
# evaluate_policy(model,env,n_eval_episodes=10,render=True)
#
# env.close()

# env_name = 'CarRacing-v0'
# env = gym.make(env_name)
# env = DummyVecEnv([lambda:env])
# PPO_path = os.path.join('Training','Saved Models','PPO_RacingCar_Model')
# model = PPO.load(PPO_path,env)
# evaluate_policy(model,env,n_eval_episodes=10,render=True)
# env.close()

env = project3_RL_custum_env.ShowerEnv()
shower_path = os.path.join('Training','Saved Models','Shower_Model_PPO')
model = PPO.load(shower_path,env)
print(evaluate_policy(model,env,n_eval_episodes=10))