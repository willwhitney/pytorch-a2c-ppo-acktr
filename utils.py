import torch
import torch.nn as nn

import copy
import json

# Necessary for my KFAC implementation.
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


def update_current_obs(obs, current_obs, obs_shape, num_stack):
    shape_dim0 = obs_shape[0]
    obs = torch.from_numpy(obs).float()
    if num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs

def serialize_opt(opt):
    # import ipdb; ipdb.set_trace()
    cleaned_opt = copy.deepcopy(vars(opt))
    return json.dumps(cleaned_opt, indent=4, sort_keys=True)

def write_options(opt, location):
    with open(location + "/opt.json", 'w') as f:
        serial_opt = serialize_opt(opt)
        print(serial_opt)
        f.write(serial_opt)
        f.flush()

