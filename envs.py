import os
import random
import math
import torch 

import gym
import numpy as np
import numpy.random as npr
from gym.spaces.box import Box
from gym import spaces
from gym.envs.registration import register
from gym.wrappers.time_limit import TimeLimit
import gym.envs.mujoco

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

import sys
sys.path.insert(0, '../action-embedding')
import gridworld.grid_world_env


def make_env(env_id, seed, rank, log_dir, add_timestep, allow_reset=False):
    def _thunk():
        # gridworld_steps = 800 if lookup is None else 100
        # register(
        #     id='GridWorld-v0',
        #     entry_point='gridworld.grid_world_env:GridWorldEnv',
        #     # max_episode_steps=gridworld_steps,
        # )

        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        if 'GridWorld' in env_id:
            env = TimeLimit(env, max_episode_steps=800)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        obs_shape = env.observation_space.shape
        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)


        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                        allow_early_resets=allow_reset)

        # if lookup is not None:
        #     env = EmbeddedAction(env, lookup, scale, cdf)

        if is_atari:
            env = wrap_deepmind(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = WrapPyTorch(env)

        return env

    return _thunk

# class EmbeddedAction(gym.Wrapper):
#     def __init__(self, env, lookup, scale, cdf=False):
#         """Return only every `skip`-th frame"""
#         gym.Wrapper.__init__(self, env)
#         # most recent raw observations (for max pooling across time steps)
#         self._lookup = lookup
#         self.low = np.stack(lookup.keys).min()
#         self.high = np.stack(lookup.keys).max()
#         self.scale = scale * max(abs(self.low), abs(self.high))
#         self.action_space = spaces.Box(
#                 -1, 1, dtype=np.float32,
#                 shape=lookup.keys[0].shape,)
#         print("Keys range: [{:.3f}, {:.3f}]".format(self.low, self.high))
#         all_base_actions = torch.stack(lookup.values)
#         print("Actions range: [{:.3f}, {:.3f}]".format(
#             all_base_actions.min(), all_base_actions.max()))

#     def step(self, action):
#         """Repeat action, sum reward, and max over last observations."""
#         total_reward = 0.0
#         done = None
#         result_obs = None

#         if hasattr(self._lookup, 'embedded_offset'):
#             offset = self._lookup.embedded_offset 
#         else:
#             offset = np.zeros(action.shape)
#         # import ipdb; ipdb.set_trace()
#         action = action * self.scale + offset
#         base_actions = self._lookup[action]
#         # print(base_actions.min().item(), base_actions.max().item())

#         # summed_actions = base_actions.sum(0)
#         # delta_x = summed_actions[1] - summed_actions[3]
#         # delta_y = summed_actions[2] - summed_actions[0]
#         # print((delta_x, delta_y))

#         # print(action)

#         for i in range(len(base_actions)):
#             # import ipdb; ipdb.set_trace()
#             action = base_actions[i]
#             if isinstance(self.env.action_space, spaces.Discrete):
#                 action = random.choices(range(len(action)), weights=action)[0]
#             obs, reward, done, info = self.env.step(np.array(action))
#             result_obs = obs
#             total_reward += reward
#             if done:
#                 break

#         return result_obs, total_reward, done, info

#     def reset(self, **kwargs):
#         return self.env.reset(**kwargs)



class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class SparseReacherEnv(gym.envs.mujoco.ReacherEnv):
    def __init__(self, *args, **kwargs):
        self.startup = True
        super().__init__(*args, **kwargs)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # import ipdb; ipdb.set_trace()
        reward = 100 if abs(reward_dist) < 0.01 else reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = (reward > 0) and not self.startup
        
        self.startup = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

register(
    id='SparseReacher-v2',
    entry_point='envs:SparseReacherEnv',
    max_episode_steps=50,
)

class SuperSparseReacherEnv(gym.envs.mujoco.ReacherEnv):
    def __init__(self, *args, **kwargs):
        self.startup = True
        super().__init__(*args, **kwargs)

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        # import ipdb; ipdb.set_trace()
        # print(reward_dist)
        reward = 100 if abs(reward_dist) < 0.001 else reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = (reward > 0) and not self.startup
        
        self.startup = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

register(
    id='SuperSparseReacher-v2',
    entry_point='envs:SuperSparseReacherEnv',
    max_episode_steps=50,
)

