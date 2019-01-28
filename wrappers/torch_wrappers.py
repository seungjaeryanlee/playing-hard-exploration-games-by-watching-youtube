import gym
from gym import spaces
import numpy as np
import torch


class TorchTensorWrapper(gym.Wrapper):
    """
    OpenAI Environment Wrapper that changes output types of `env.reset()` and
    `env.step()` to `torch.Tensor`.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        ob = self.env.reset()
        ob = torch.FloatTensor([ob])
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        return ob, reward, done, info


class TorchPermuteWrapper(gym.ObservationWrapper):
    """
    OpenAI Atari Environment Wrapper that permutes environment
    observation to PyTorch style: NCHW.
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(shp[2], shp[0], shp[1]), dtype=np.float32
        )

    def observation(self, observation):
        return observation.permute(0, 3, 1, 2)


def wrap_pytorch(env):
    """
    Wrap environment to be compliant to PyTorch agents.
    """
    env = TorchTensorWrapper(env)
    env = TorchPermuteWrapper(env)

    return env
