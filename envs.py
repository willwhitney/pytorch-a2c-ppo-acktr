import os
import numpy as np
import ipdb

import gym
from gym.spaces.box import Box
from gym.envs.registration import register
import gym.envs.mujoco

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
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=True)
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

class VisibleHopperEnv(gym.envs.mujoco.HopperEnv):
    def _get_obs(self):
        # import pdb; pdb.set_trace()
        return np.concatenate([
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    # def step(self, a):
    #     posbefore = self.sim.data.qpos[0]
    #     self.do_simulation(a, self.frame_skip)
    #     posafter, height, ang = self.sim.data.qpos[0:3]
    #     alive_bonus = 1.0
    #     reward = (posafter - posbefore) / self.dt
    #     reward += alive_bonus
    #     reward -= 1e-3 * np.square(a).sum()
    #     # reward_components = np.array([abs((posafter - posbefore) / self.dt),
    #     #                               alive_bonus,
    #     #                               1e-3 * np.square(a).sum()])
    #     # print(list(reward_components / reward_components.sum()))
    #     s = self.state_vector()
    #     done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
    #                 (height > .7) and (abs(ang) < .2))
    #     ob = self._get_obs()
    #     # if reward > 1.5:
    #     #     import pdb; pdb.set_trace()
    #     return ob, reward, done, {}

register(
    id='VisibleHopper-v2',
    entry_point='envs:VisibleHopperEnv',
)

class VisibleSwimmerEnv(gym.envs.mujoco.SwimmerEnv):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

register(id='VisibleSwimmer-v2', entry_point='envs:VisibleSwimmerEnv')
