import numpy as np

import torch
from torch.autograd import Variable

from gymnasium.spaces import Box

from MARL.utils.dictionary import AttrDict

from MARL.algorithms.maddpg_dev import MADDPG
from MARL.utils.buffer import ReplayBuffer
from MARL.models.abstractwrapper import AbstractWrapper

class MADDPGWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        #possible algs: MADDPG,DDPG
        #possible adversary algs: MADDPG,DDPG  
        self.model = MADDPG.init_from_env(*args, **kwargs)

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')

        self.replay_buffer = ReplayBuffer(int(kwargs['buffer_length']), self.model.nagents,
                                    [self.env.observation_space(agent).shape[0] for agent in self.env.possible_agents], # comment this line for fix size
                                    [self.env.action_space(agent).shape[0] if isinstance(self.env.action_space(agent), Box) 
                                    else self.env.action_space(agent).n for agent in self.env.possible_agents])
        
        self.t = 0

    def step(self, obs_dict, *args, **kwargs):

        # Convert the observations to torch Tensor
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

        # Model step
        torch_agent_actions = self.model.step(torch_obs, explore=self.config['explore'])
        # clip the action between a minimum and maximum value to prevent noisy warning messages
        agent_actions = {agent: np.clip(ac, 0, 1) for agent, ac in zip(self.env.possible_agents, torch_agent_actions)}
        return agent_actions

    def save(self, fname):
        self.model.save(fname)

    def begin_episode(self, ep_i, *args, **kwargs):
        self.model.prep_rollouts(device='cpu')
        #noise w.r.t. episode percent remaining
        explr_pct_remaining = max(0, self.config.n_exploration_eps - ep_i) / self.config.n_exploration_eps
        self.model.scale_noise(self.config.final_noise_scale + (self.config.init_noise_scale - self.config.final_noise_scale) * explr_pct_remaining)
        self.model.reset_noise()

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones = args
        self.t += self.config.n_rollout_threads

        self.replay_buffer.push(obs_dict, agent_actions, rewards, next_obs, dones)  # Use obs_dict here instead of obs
        if (len(self.replay_buffer) >= self.config.batch_size and
            (self.t % self.config.steps_per_update) < self.config.n_rollout_threads):
            #train critic, actor, target networks
            if self.device == 'cuda':
                self.model.prep_training(device='gpu')
            else:
                self.prep_training(device='cpu')
            for u_i in range(self.config.n_rollout_threads):
                for a_i in range(self.model.nagents):
                    sample = self.replay_buffer.sample(self.config.batch_size, to_gpu=self.device)
                    self.model.update(sample, a_i, logger=None)
                self.model.update_all_targets()
            self.model.prep_rollouts(device='cpu')

