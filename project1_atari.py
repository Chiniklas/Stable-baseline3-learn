# 1.import dependencies
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import ROMS

# 2.test environment
# !python -m atari_py.import_roms .\ROMS\ROMS
environment_name = 'Breakout-v0'
env = gym.make(environment_name)
# env.reset()
# print(env.action_space)
# print(env.observation_space)

episodes = 10

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



# model = A2C('CnnPolicy',env,verbose=1,device='cuda')
# model.learn(total_timesteps=20000)
#
# for episode in range(1,episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action,_ = model.predict(obs)
#         obs,reward,done,info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode,score))
# env.close()

# 3.vectorise environment and train model
# train multiple models parallely
env = make_atari_env('Breakout-v0',n_envs=4,seed=0)
env = VecFrameStack(env,n_stack=4)
print(env.reset())
log_path = os.path.join('Training','Logs')
model = A2C('CnnPolicy',env,verbose=1,tensorboard_log=log_path,device='cuda')
model.learn(total_timesteps=100000)
# 4.save and reload model
a2c_path = os.path.join('Training','Saved Models','A2C_Breakout_Model')
model.save(a2c_path)
del model
model = A2C.load(a2c_path,env)
# 5.evaluate and test
# we can only evaluate on a single env, but we vectorised it before
env = make_atari_env('Breakout-v0',n_envs=1,seed=0)
env = VecFrameStack(env,n_stack=4)
evaluate_policy(model,env,n_eval_episodes=10,render=True)

