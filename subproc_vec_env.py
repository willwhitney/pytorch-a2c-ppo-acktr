import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

def worker(remote, parent_remote, env_fn_wrapper):
    # import sys
    # sys.stdout = lambda _: None
    # sys.stderr = sys.stdout

    parent_remote.close()
    env = env_fn_wrapper.x()
    env._subproc_did_finish = False

    def steps(actions):
        obs, done, info = (None for _ in range(3))
        reward = 0
        # steps = 0
        # print("actions len: {}".format(len(actions)))
        for action in actions:
            obs, r, done, info = env.step(action)
            reward += r
            # steps += 1
            if done:
                break
        # print("steps: {}".format(steps))
        return obs, reward, done, info

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            if env._subproc_did_finish:
                env._subproc_did_finish = False
                remote.send((env.reset(), 0, False, {}))
            else:
                ob, reward, done, info = env.step(data)
                if done:
                    env._subproc_did_finish = True
                else:
                    env._subproc_did_finish = False
                remote.send((ob, reward, done, info))
        elif cmd == 'steps':
            remote.send(steps(data))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))

        elif cmd == 'set_state':
            pos = data[:env.unwrapped.model.nq]
            vel = data[env.unwrapped.model.nq: 
                       env.unwrapped.model.nq + env.unwrapped.model.nv]
            env.unwrapped.set_state(pos, vel)
            remote.send(env.unwrapped.state_vector())
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, 
                           args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in 
                zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True 
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def steps(self, action_seqs):
        n = min(len(self.remotes), len(action_seqs))
        for i in range(n):
            remote, actions = self.remotes[i], action_seqs[i]
            remote.send(('steps', actions))

        results = [self.remotes[i].recv() for i in range(n)]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos


    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def set_state(self, states):
        self.reset()
        n = min(len(self.remotes), len(states))
        for i in range(n):
            remote, state = self.remotes[i], states[i]
            remote.send(('set_state', state))
        return np.stack([self.remotes[i].recv() for i in range(n)])
