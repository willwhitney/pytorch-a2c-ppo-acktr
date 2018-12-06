import numpy as np
import numpy.random as npr
import torch

class DummyLookup:
    def __init__(self, action_space, traj_len):
        # import ipdb; ipdb.set_trace()
        self.keys = [action_space.low, action_space.high]
        self.action_space = action_space
        self.traj_len = traj_len

    def query_fn(self, key):
        # outcome = (self.action_space.high - self.action_space.low) * key
        action_sequence = [np.copy(key) for _ in range(self.traj_len)]
        return torch.stack([torch.from_numpy(a) for a in action_sequence])

    def __getitem__(self, key):
        return self.query_fn(key)

class SpikyDummyLookup(DummyLookup):
    def query_fn(self, key):
        # scale = self.action_space.high - self.action_space.low
        steps_sum = key * self.traj_len
        action_sequence = []
        for i in range(self.traj_len):
            steps_after = self.traj_len - (i + 1)
            least_after = steps_after * self.action_space.low
            most_after = steps_after * self.action_space.high
            upper_bound = steps_sum - sum(action_sequence) - least_after
            lower_bound = steps_sum - sum(action_sequence) - most_after

            lower_bound = np.maximum(self.action_space.low, lower_bound)
            upper_bound = np.minimum(self.action_space.high, upper_bound)
            action = npr.uniform(lower_bound, upper_bound)
            action_sequence.append(action)
        return torch.stack([torch.from_numpy(a) for a in action_sequence])

if __name__ == "__main__":
    from gym import spaces
    space = spaces.Box(low=-np.ones(2), high=np.ones(2))
    lookup = SpikyDummyLookup(space, 2)
    zero_answer = lookup[np.array([0,0])]
    import ipdb; ipdb.set_trace()