"""
Currently all wrappers are for Bounce

Modified from https://github.com/openai/coinrun/blob/master/coinrun/wrappers.py
"""

import gym
import numpy as np


class EpsilonGreedyWrapper(gym.Wrapper):
    """
    Wrapper to perform a random action each step instead of the requested action,
    with the provided probability.
    """

    def __init__(self, env, prob=0.05):
        gym.Wrapper.__init__(self, env)
        self.prob = prob
        self.num_envs = env.num_envs

    def reset(self):
        return self.env.reset()

    def step(self, action):
        if np.random.uniform() < self.prob:
            action = np.random.randint(self.env.action_space.n, size=self.num_envs)

        return self.env.step(action)
