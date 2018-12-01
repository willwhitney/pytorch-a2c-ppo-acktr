import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.envs.registration import register
import mujoco_py
from gym import error, spaces
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class LinearPointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        # self.action_space = spaces.Box(low=-np.ones(2), high=np.ones(2), dtype=np.float32)
        # self.model.nu = 2
        mujoco_env.MujocoEnv.__init__(self, dir_path + "/assets/point_mass.xml", 4)
        utils.EzPickle.__init__(self)


    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        reward = - np.linalg.norm(self.sim.data.qpos)
        ob = self._get_obs()
        return ob, reward, False, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos, qvel]).ravel()

    def do_simulation(self, ctrl, n_frames):
        self.set_state(self.sim.data.qpos, ctrl)
        self.sim.data.ctrl[:] = ctrl

        for _ in range(n_frames):
            self.sim.step()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()


register(
    id='LinearPointMass-v0',
    entry_point='pointmass.point_mass:LinearPointMassEnv',
    max_episode_steps=100,
)
