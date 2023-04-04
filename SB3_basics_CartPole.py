import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold
from stable_baselines3 import DQN

####################### load environment ########################
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

# print(environment_name)

####################### Understanding the environment ########################
# episodes = 5
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

# action space 可能采取的行动
# print(env.action_space)

# observation space 是接受到的观测值？？
# print(env.observation_space)

#################################Train an agent################################
log_path = os.path.join('Training','Logs')
# print(log_path)
env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])

# model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
# model.learn(total_timesteps=20000)
#
# ############################### Save and Reload Model####################################
# PPO_PATH = os.path.join('Training','Saved Models','PPO_Model_CartPole')
# model.save(PPO_PATH)
# del model
# model = PPO.load(PPO_PATH,env=env)
# model.learn(total_timesteps=1000)
#
# ############################## test and evaluation #########################################
# evaluate_policy(model,env,n_eval_episodes=10,render=True)
#
# env.close()

# deploy the model in simulation
# episodes = 5
# for episode in range(1,episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0
#     while not done:
#         env.render()
#         action,_ = model.predict(obs) # now using model(learned optimal policy) here
#         obs,reward,done,info = env.step(action)
#         score += reward
#     print('Episode:{} Score:{}'.format(episode,score))
# env.close()


############################### viewing logs in Tensorboard########################
training_log_path = os.path.join(log_path,'PPO_14')
# use anaconda prompt to show the metrics
# cd to the log directory
# tensorboard --logdir=.
# then open the local host in chrome

# core metrics
# 1. average reward
# 2. average episode length
# optuna to tune hyperparameter

##############################adding a callback to the training stage#########################
# gives you the automated capability to stop the training when the agent hit a reward threshold
save_path = os.path.join('Training','Saved Models')
stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200,verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1,
                             )
# model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
# model.learn(total_timesteps=20000,callback=eval_callback)

#################################changing policies##############################
# changing neural network architecture
# this can get very very complex
net_arch = [dict(pi=[128,128,128,128],vf=[128,128,128,128])]
# model = PPO('MlpPolicy',env,verbose=1,tensorboard_log=log_path,policy_kwargs={'net_arch':net_arch})
# model.learn(total_timesteps=20000,callback=eval_callback)

#############################using an Alternate Algorithm##################################

model = DQN('MlpPolicy',env,verbose=1,tensorboard_log=log_path)
model.learn(total_timesteps=20000)

# how to save and reload
# model.save()
# DQN.load()
