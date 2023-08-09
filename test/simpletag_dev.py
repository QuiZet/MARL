import argparse
import torch
import torch.nn.functional as F
import time
import os
import numpy as np
from gymnasium.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import sys
sys.path.append('./')
sys.path.append('../MARL/')

from MARL.utils.buffer import ReplayBuffer
from MARL.algorithms.maddpg_dev import MADDPG
import pygame
from pettingzoo.mpe import simple_tag_v3

USE_CUDA = False  # torch.cuda.is_available()

def run(config):
    # Logging
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) 
                         for folder in model_dir.iterdir() if str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    # Seeding
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    # Training
    env = simple_tag_v3.parallel_env(render_mode='rgb', num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=True)
    obs_dict, _ = env.reset()  # Get the initial observations and ignore the second return value

    # a: [16, 16, 16, 14], observation spaces of agents
    a = [env.observation_space(agent).shape[0] for agent in env.possible_agents]
    print(f'a:{a}')
    # b: [5, 5, 5, 5], action spaces of agents
    b = [env.action_space(agent).shape[0] if isinstance(env.action_space(agent), Box) else env.action_space(agent).n for agent in env.possible_agents]
    print(f'b:{b}')
    
    #possible algs: MADDPG,DDPG
    #possible adversary algs: MADDPG,DDPG  
    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim)

    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [env.observation_space(agent).shape[0] for agent in env.possible_agents],
                                 [env.action_space(agent).shape[0] if isinstance(env.action_space(agent), Box) else env.action_space(agent).n for agent in env.possible_agents])

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))
        obs = env.reset()
        print(f'obs:{obs}')
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        #noise w.r.t. episode percent remaining
        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # Episode length
        for et_i in range(config.episode_length):
            print(f'config.episode_length:{et_i} {config.episode_length}')

            # Convert the observations to torch Tensor
            torch_obs = []
            for agent in env.possible_agents:
                #print(f'agent:{agent}')
                agent_obs = [obs_dict[agent]]
                if agent == 'agent_0':
                    agent_obs = np.insert(agent_obs, 14, 0, axis=-1)
                    agent_obs = np.insert(agent_obs, 0, 0, axis=-1)
                    agent_obs = [agent_obs]
                print(f'agent_obs:{agent_obs}')
                torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))
                print(f'Tensor agent_obs:{agent_obs}')

            #torch_obs = torch.cat(torch_obs, dim=1)  # Concatenate observations

            # Get actions from the MADDPG policy and explore if needed
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            for agent, ac in zip(env.possible_agents, torch_agent_actions):
                print(f'agent:{agent} ac:{type(ac)}')
            agent_actions = {agent: ac for agent, ac in zip(env.possible_agents, torch_agent_actions)}
            print(f'agent_actions:{agent_actions}')
            # Take a step in the environment with the selected actions
            next_obs, rewards, dones, truncations, infos = env.step(agent_actions)
            print(f'next_obs:{next_obs}')

            replay_buffer.push(obs_dict, agent_actions, rewards, next_obs, dones)  # Use obs_dict here instead of obs
            obs_dict = next_obs  # Update obs_dict for the next iteration
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                #train critic, actor, target networks
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size, to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')

        ep_rews = replay_buffer.get_average_rewards(config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            maddpg.save(run_dir / 'model.pt')

    maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    return env, maddpg

def print_observation_space_dimensions(env):
    for agent in env.possible_agents:
        obsp = env.observation_space(agent)
        print(f"Agent {agent}: {obsp.shape}")
def print_policy_network_dimensions(maddpg):
    for i, agent in enumerate(maddpg.agents):
        print(f"Agent {i + 1} - Policy Network:")
        print(f"Input Dimension: {agent.policy.fc1.in_features}")
        print(f"Output Dimension: {agent.policy.fc3.out_features}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of directory to store model/training contents")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')

    config = parser.parse_args()

    env, maddpg = run(config)
    print_observation_space_dimensions(env)
    print_policy_network_dimensions(maddpg)
