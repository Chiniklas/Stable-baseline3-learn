from gym import Env
from gym.spaces import Discrete,Box
import numpy as np
import random

class ShowerEnv(Env):
    def __init__(self):
        # setting action space
        self.action_space = Discrete(3)
        # temperature array
        self.observation_space = Box(low = np.array([0]),high=np.array([100]))
        # set start temp
        self.state = 38 + random.randint(-3,3)
        # set shower length
        self.shower_length = 60

    def step(self, action):
        # apply action
        self.state += action - 1
        self.shower_length -= 1

        # calculate reward
        if self.state >= 37 and self.state <= 39:
            reward =1
        else:
            reward = -1

        # check if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # apply temperature noise
        self.state += random.randint(-1,1)

        # set placeholder for info
        info = {}

        # return step information
        return self.state,reward,done,info


    def render(self):
        # implement visualization
        pass

    def reset(self):
        # reset shower temp
        self.state = 38 + random.randint(-3,3)
        # reset shower length
        self.shower_length = 60
        return self.state
