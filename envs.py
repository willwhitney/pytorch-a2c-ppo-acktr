import os
import numpy as np

import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import pybullet_envs
    import roboschool
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, repeat):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        env = RepeatEnv(env, skip=repeat)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        if is_atari:
            env = wrap_deepmind(env)
        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]]
        )

    def _observation(self, observation):
        return observation.transpose(2, 0, 1)

class RepeatEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        result_obs = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 1:
                result_obs = obs
            total_reward += reward
            if done:
                break

        return result_obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
