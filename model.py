import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_
from gym import spaces

import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space,
                 real_variance=False,
                 tanh_mean=False,
                 base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], **base_kwargs)
            # self.base = MLPBase(
            #         obs_shape[0] * obs_shape[1] * obs_shape[2],
            #         **base_kwargs)
        elif len(obs_shape) == 1:
            self.base = MLPBase(obs_shape[0], **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs,
                                     real_variance=real_variance,
                                     tanh_mean=tanh_mean)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def reset(self, *args):
        pass


class EmbeddedPolicy(Policy):
    def __init__(self, obs_shape, action_space, lookup=None, decoder=None,
                 scale=0.1, neighbors=1, cdf=False, **kwargs):
        if lookup:
            self.embedded_action_size = lookup.keys[0].size(0)
        else:
            self.embedded_action_size = decoder.layers[0].in_features

        super().__init__(obs_shape,
                action_space=spaces.Box(-1, 1, shape=(self.embedded_action_size,)),
                **kwargs)
        self.lookup = lookup
        self.decoder = decoder


        if self.lookup:
            self.low = np.stack(self.lookup.keys).min()
            self.high = np.stack(self.lookup.keys).max()
            self.scale = scale * max(abs(self.low), abs(self.high))
        else:
            self.scale = scale

        self.neighbors = neighbors
        self.cdf = cdf
        self.pending_plans = None

    def act(self, *args, **kwargs):
        values, e_actions, _, e_logprobs, rnn_hxs = super().act(*args, **kwargs)
        if self.lookup:
            scaled_e_actions = [self.scale_key(a.cpu()) for a in e_actions]
            plans = torch.stack(
                    [self.lookup.minnorm_match(a, neighbors=self.neighbors)
                     for a in scaled_e_actions])
        else:
            plans = self.decoder(e_actions * self.scale)

        if self.pending_plans is None:
            self.pending_plans = [[] for _ in range(len(e_actions))]

        action_logprobs = e_logprobs.detach()
        for i in range(len(e_actions)):
            if len(self.pending_plans[i]) == 0:
                self.pending_plans[i] = list(plans[i])
            else:
                # setting the logprob to infinity means that there will be no 
                # update to the model based on this action
                action_logprobs[i] = np.infty

        base_actions = torch.stack([plan[0] for plan in self.pending_plans])
        self.pending_plans = [plan[1:] for plan in self.pending_plans]
        return (
                values,
                e_actions,
                base_actions,
                action_logprobs,
                rnn_hxs
            )


    def scale_key(self, key):
        if self.lookup and hasattr(self.lookup, 'embedded_offset'):
            offset = self.lookup.embedded_offset
        else:
            offset = torch.zeros(key.size())

        key = key * self.scale + offset
        return key.numpy()


    def reset(self, dones):
        for i, done in enumerate(dones):
            if self.pending_plans is not None:
                if done > 0: self.pending_plans[i] = []


class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.num_inputs = num_inputs

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m,
            init_normc_,
            lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs.view(-1, self.num_inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
