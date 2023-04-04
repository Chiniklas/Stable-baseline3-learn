# 1.import Dependencies
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import time

# 2.Test Environment
env_name = 'CarRacing-v0'
env = gym.make(env_name)
# print(env.reset())
# print(env.action_space)
# print(env.observation_space)
env.reset()
env.render()
# time.sleep(10)
env.close()

# episodes = 5
#
# for episode in range(1,episodes+1):
#     state = env.reset()
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state,reward,done,info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode,score))
# env.close()

# 3.Train Model
env = gym.make(env_name)
env = DummyVecEnv([lambda:env])
log_path = os.path.join('Training','Logs')
model = PPO('CnnPolicy',env,verbose=1,tensorboard_log=log_path,device='cuda')
model.learn(total_timesteps=200000)
# 4.Save Model
PPO_path = os.path.join('Training','Saved Models','PPO_RacingCar_Model')
model.save(PPO_path)

# 5.Evaluate and Test
del model
model = PPO.load(PPO_path,env)
evaluate_policy(model,env,n_eval_episodes=10,render=True)
env.close()


