import copy
import glob
import os
import time
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from arguments import get_args
from envs import make_env
from model import Policy, EmbeddedPolicy
from storage import RolloutStorage
import utils
from utils import update_current_obs, write_options
from visualize import visdom_plot
import algo

from pointmass import point_mass
from dummy_lookup import DummyLookup, ReverseDummyLookup, SpikyDummyLookup

# from pyvirtualdisplay import Display
# display_ = Display(visible=0, size=(550,550))
# display_.start()

# sys.path.insert(0, '../action-embedding')
# import gridworld.grid_world_env
# import gridworld
args = get_args()

assert args.algo in ['a2c', 'ppo', 'acktr']
if args.recurrent_policy:
    assert args.algo in ['a2c', 'ppo'], \
        'Recurrent policy is not implemented for ACKTR'

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

embed_flag = "raw" if args.action_embedding is None else "embed"
args.log_dir = os.path.join(args.log_dir, args.env_name, embed_flag, args.name)

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
    for f in files:
        os.remove(f)

write_options(args, args.log_dir)

def construct_envs(log_dir=args.log_dir, allow_reset=False):
    envs = [make_env(
                args.env_name, args.seed, i, log_dir,
                args.add_timestep,
                allow_reset=allow_reset)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    return envs


def main():
    print("#######")
    print(("WARNING: All rewards are clipped or normalized so you need "
        "to use a monitor (see envs.py) or visdom plot to get true rewards"))
    print("#######")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.vis:
        from visdom import Visdom
        viz = Visdom(server='http://100.97.69.42', port=args.port)
        win = None

    envs = construct_envs()

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    # render_envs = construct_envs(log_dir='/tmp', allow_reset=True)

    lookup = None
    if args.dummy_embedding:
        if args.dummy_embedding == 'spiky':
            lookup = SpikyDummyLookup(envs.action_space, args.dummy_traj_len)
        elif args.dummy_embedding == 'reverse':
            lookup = ReverseDummyLookup(envs.action_space, args.dummy_traj_len)
        else:
            lookup = DummyLookup(envs.action_space, args.dummy_traj_len)
        actor_critic = EmbeddedPolicy(obs_shape, envs.action_space,
                lookup=lookup,
                scale=args.scale,
                real_variance=args.real_variance,
                base_kwargs={'recurrent': args.recurrent_policy})

    elif args.action_decoder is not None:
        if 'PointMass' in args.env_name: base_env_name = 'LinearPointMass-v0'
        else: base_env_name = args.env_name.strip("Super").strip("Sparse")
        decoder = torch.load(
                "../action-embedding/results/{}/{}/decoder.pt".format(
                base_env_name,
                args.action_decoder))
        actor_critic = EmbeddedPolicy(obs_shape, envs.action_space,
                decoder=decoder,
                scale=args.scale,
                neighbors=args.neighbors,
                cdf=args.cdf,
                real_variance=args.real_variance,
                tanh_mean=args.tanh_mean,
                base_kwargs={'recurrent': args.recurrent_policy})

    elif args.action_embedding is not None:
        lookup = torch.load(
                "../action-embedding/results/{}/{}/lookup.pt".format(
                args.env_name.strip("Super").strip("Sparse"),
                args.action_embedding))
        actor_critic = EmbeddedPolicy(obs_shape, envs.action_space,
                lookup=lookup,
                scale=args.scale,
                neighbors=args.neighbors,
                cdf=args.cdf,
                real_variance=args.real_variance,
                tanh_mean=args.tanh_mean,
                base_kwargs={'recurrent': args.recurrent_policy})

    else:
        actor_critic = Policy(obs_shape, envs.action_space,
                real_variance=args.real_variance,
                tanh_mean=args.tanh_mean,
                base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if isinstance(actor_critic, EmbeddedPolicy):
        action_shape = actor_critic.embedded_action_size
    else:
        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, lr=args.lr,
                               eps=args.eps, alpha=args.alpha,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(actor_critic, args.clip_param, args.ppo_epoch, args.num_mini_batch,
                         args.value_loss_coef, args.entropy_coef, lr=args.lr,
                               eps=args.eps,
                               max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(actor_critic, args.value_loss_coef,
                               args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape,
        envs.action_space, action_shape, actor_critic.recurrent_hidden_state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    obs = envs.reset()
    update_current_obs(obs, current_obs, obs_shape, args.num_stack)
    rollouts.obs[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    current_obs = current_obs.to(device)
    rollouts.to(device)

    start = time.time()
    for j in range(num_updates):
        actions = []

        render_iteration = (j > 0) and (j % args.render_interval == 0)
        if render_iteration:
            images = [envs.render(mode='rgb_array')]
        for step in range(args.num_steps):

            # Sample actions
            with torch.no_grad():
                value, e_action, action, action_log_prob, recurrent_hidden_states \
                    = actor_critic.act(
                        rollouts.obs[step],
                        rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            actions.append(action)
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            if render_iteration:
                images.append(envs.render(mode='rgb_array'))


            actor_critic.reset(done)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            masks = masks.to(device)

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, current_obs, obs_shape, args.num_stack)
            rollouts.insert(current_obs, recurrent_hidden_states, e_action,
                    action_log_prob, value, reward, masks)

        if render_iteration:
            utils.save_gif('{}/{}.mp4'.format(args.log_dir, j),
                            [torch.tensor(im).float()/255 for im in images],
                            color_last=True)

        actions = torch.cat(actions, 0)
        # print(actions.min().item(), actions.max().item(), flush=True)
        # mean_action = torch.cat(actions, 0).mean(0)
        # print("({:.3f}, {:.3f})".format(mean_action[0], mean_action[1]))

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.obs[-1],
                                                rollouts.recurrent_hidden_states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j > 0 and j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss), flush=True)
        if args.vis and j % args.vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, args.log_dir, args.log_dir,
                                  args.algo, args.num_frames)
            except IOError:
                pass

        # if j % args.render_interval == 0:
        #     if isinstance(envs, VecNormalize):
        #         render_envs.ob_rms.mean = np.copy(envs.ob_rms.mean)
        #         render_envs.ob_rms.var = np.copy(envs.ob_rms.var)
        #         render_envs.ret_rms.mean = np.copy(envs.ret_rms.mean)
        #         render_envs.ret_rms.var = np.copy(envs.ret_rms.var)

        #     render_rollouts = RolloutStorage(1000, args.num_processes, obs_shape,
        #         envs.action_space, actor_critic.recurrent_hidden_state_size)

        #     render_current_obs = torch.zeros(args.num_processes, *obs_shape)
        #     obs = render_envs.reset()
        #     update_current_obs(obs, render_current_obs, obs_shape, args.num_stack)
        #     render_rollouts.obs[0].copy_(current_obs)
        #     render_current_obs = render_current_obs.to(device)
        #     render_rollouts.to(device)

        #     images = [render_envs.render(mode='rgb_array')]
        #     for step in range(500):
        #         with torch.no_grad():
        #             # import ipdb; ipdb.set_trace()
        #             value, e_action, action, action_log_prob, recurrent_hidden_states \
        #                 = actor_critic.act(
        #                     render_rollouts.obs[step],
        #                     render_rollouts.recurrent_hidden_states[step],
        #                     render_rollouts.masks[step])

        #             cpu_actions = action.squeeze(1).cpu().numpy()

        #             # Obser reward and next obs
        #             obs, reward, done, info = render_envs.step(cpu_actions)
        #             actor_critic.reset(done)
        #             images.append(render_envs.render(mode='rgb_array'))

        #             if render_current_obs.dim() == 4:
        #                 render_current_obs *= masks.unsqueeze(2).unsqueeze(2)
        #             else:
        #                 render_current_obs *= masks

        #             update_current_obs(obs, render_current_obs, obs_shape, args.num_stack)

        #     try:
        #         utils.save_gif('{}/{}.mp4'.format(args.log_dir, j),
        #                 [torch.tensor(im).float()/255 for im in images],
        #                 color_last=True)
        #     except:
        #         import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    main()
