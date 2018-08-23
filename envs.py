import os
import numpy as np
import ipdb

import gym
import numpy as np
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
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

def make_env(env_id, seed, rank, log_dir, repeat, add_timestep, channel_width, add_timeout):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)
        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)

        if isinstance(env.unwrapped, TerminatingVisibleSwimmerEnv):
            env.unwrapped.channel_width = channel_width

        env = RepeatEnv(env, skip=repeat)

        obs_shape = env.observation_space.shape
        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)
 
        if add_timeout and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimeout(env)

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

class AddTimeout(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimeout, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        if self.env.env._elapsed_steps >= self.env.env._max_episode_steps - 1:
            timeout = 1
        else:
            timeout = 0
        return np.concatenate((observation, [timeout]))

class DatasetMaker(gym.ObservationWrapper):
    def __init__(self, env):
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # self.reward_range = self.env.reward_range
        # self.metadata = self.env.metadata
        self.collecting_data = False
        self.dataset = []
        self.current_traj = [[], []]
        import ipdb; ipdb.set_trace()

    def observation(self, observation):
        return observation



    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        timeout = observation[:, -1]

        # for i in range(observation.shape[0]):
        #     if done[i] == 1:


        return self.observation(observation[:, :-1]), reward, done, info

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
        return np.concatenate([
            self.sim.data.qpos.flat,
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

register(
    id='VisibleHopper-v2',
    entry_point='envs:VisibleHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

class VisibleSwimmerEnv(gym.envs.mujoco.SwimmerEnv):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

register(
    id='VisibleSwimmer-v2',
    entry_point='envs:VisibleSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)


class TerminatingVisibleSwimmerEnv(gym.envs.mujoco.SwimmerEnv):
    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        yposafter = self.sim.data.qpos[1]
        if hasattr(self, 'channel_width'):
            done = not (-self.channel_width/2 < yposafter < self.channel_width/2)
        else:
            done = False
        return ob, reward, done, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

register(
    id='TerminatingVisibleSwimmer-v2',
    entry_point='envs:TerminatingVisibleSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)