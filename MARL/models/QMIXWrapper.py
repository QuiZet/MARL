import copy
import numpy as np

import torch
from torch.autograd import Variable

from gymnasium.spaces import Box

from MARL.utils.dictionary import AttrDict

from MARL.algorithms.qmix.qmix_smac import QMIX_SMAC
from MARL.algorithms.qmix.normalization import Normalization
from MARL.algorithms.qmix.replay_buffer import ReplayBuffer
from MARL.models.abstractwrapper import AbstractWrapper

class QMIXWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')


        # Create env
        self.args = args
        self.env_info = self.env.get_env_info()
        self.N = self.env_info["n_agents"]  # The number of agents
        self.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.N))
        print("obs_dim={}".format(self.obs_dim))
        print("state_dim={}".format(self.state_dim))
        print("action_dim={}".format(self.action_dim))
        print("episode_limit={}".format(self.episode_limit))

        # Create N agents
        self.agent_n = QMIX_SMAC(self.config)
        self.replay_buffer = ReplayBuffer(self.config)

        self.epsilon = self.config.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.total_steps = 0
        if self.config.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

        # output log dictionary
        self.log_dict_out = None

    def step(self, obs_dict, *args, **kwargs):
        # Convert the observations to torch Tensor
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

        # Each agent selects actions based on its own local observations(add noise for exploration)
        agent_actions = dict()
        for agent, obs in zip(self.agent_n, obs_dict):
            agent_actions[agent] = self.agent_n[agent].choose_action(obs_dict[obs], noise_std=self.noise_std)
        return agent_actions
    
    # Add env_evaluation in the main trainable environemnt and the option for the evaluation

    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        self.win_tag = False
        self.episode_reward = 0
        if self.config.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        self.last_onehot_a_n = np.zeros((self.N, self.action_dim))  # Last actions of N agents(one-hot)

    def end_episode(self, *args, **kwargs):
        if self.replay_buffer.current_size >= self.args.batch_size:
            self.agent_n.train(self.replay_buffer, self.total_steps)  # Training

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones = args
        self.log_dict_out = dict(rewards)

        # Store the transition
        self.replay_buffer.store_transition(obs_dict, agent_actions, rewards, next_obs, dones)
        # Decay noise_std
        if self.config.use_noise_decay:
           self.noise_std = self.noise_std - self.noise_std_decay if self.noise_std - self.noise_std_decay > self.config.noise_std_min else self.config.noise_std_min

        if self.replay_buffer.current_size > self.config.batch_size:
            # Train each agent individually
            for agent_id in self.env.possible_agents:
                actor_loss, critic_loss = self.agent_n[agent_id].train(self.replay_buffer, self.agent_n)

                if self.config.log_loss:
                    loss_dict = dict()
                    loss_dict['actor_loss' + agent_id] = actor_loss
                    loss_dict['critic_loss' + agent_id] = critic_loss
                    #print(f'loss_dict:{type(loss_dict)}')
                    #print(f'self.log_dict_out:{type(self.log_dict_out)}')
                    self.log_dict_out = self.log_dict_out | loss_dict 

    def get(self, *args, **kwargs):
        what = kwargs['what']
        if what == 'log_dict':
            return self.log_dict_out
        return None
