"""Environment wrapper to train PyTorch networks."""
from typing import Any, Tuple

import gym
import numpy as np
import torch
from gym import spaces


class TorchTensorWrapper(gym.Wrapper):
    """
    Change outputs to `torch.Tensor`.

    OpenAI Environment Wrapper that changes output types of `env.reset()` and
    `env.step()` to `torch.Tensor`.
    """

    def __init__(self, env: Any) -> None:
        gym.Wrapper.__init__(self, env)

    def reset(self) -> torch.FloatTensor:  # noqa: D102
        ob = self.env.reset()
        ob = torch.FloatTensor([ob])
        return ob

    def step(
        self, action: Any
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, dict
    ]:  # noqa: D102
        ob, reward, done, info = self.env.step(action)
        ob = torch.FloatTensor([ob])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])
        return ob, reward, done, info


class TorchPermuteWrapper(gym.ObservationWrapper):
    """
    Permute environment observation to PyTorch style.

    OpenAI Atari Environment Wrapper that permutes environment
    observation to PyTorch style: NCHW.
    """

    def __init__(self, env: Any) -> None:
        gym.ObservationWrapper.__init__(self, env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(shp[2], shp[0], shp[1]), dtype=np.float32
        )

    def observation(
        self, observation: torch.FloatTensor
    ) -> torch.FloatTensor:  # noqa: D102
        return observation.permute(0, 3, 1, 2)


def wrap_pytorch(env: Any) -> Any:
    """Wrap environment to be compliant to PyTorch agents."""
    env = TorchTensorWrapper(env)
    env = TorchPermuteWrapper(env)

    return env
