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
        self.cfgs = args[2]
        print(f'self.cfgs:{self.cfgs}')
        self.cfgs_model = self.cfgs['model']
        self.cfgs_environment = self.cfgs['environment']
        print(f'self.cfgs_model:{self.cfgs_model} type:{type(self.cfgs_model)}')
        print(f'self.cfgs_environment:{self.cfgs_environment} type:{type(self.cfgs_environment)}')

        # Create env
        env_config = dict()
        env_info = self.env.get_env_info()
        env_config['N'] = env_info["n_agents"]  # The number of agents
        env_config['obs_dim'] = env_info["obs_shape"]  # The dimensions of an agent's observation space
        env_config['state_dim'] = env_info["state_shape"]  # The dimensions of global state space
        env_config['action_dim'] = env_info["n_actions"]  # The dimensions of an agent's action space
        env_config['episode_limit'] = env_info["episode_limit"]  # Maximum number of steps per episode
        env_config['epsilon_decay'] = (self.config['epsilon'] - self.config['epsilon_min']) / self.config['epsilon_decay_steps']

        self.all_cfgs = self.cfgs_model | self.cfgs_environment | env_config
        self.all_cfgs = AttrDict(self.all_cfgs)
        print(f'self.all_cfgs:{self.all_cfgs}')

        print("number of agents={}".format(self.all_cfgs.N))
        print("obs_dim={}".format(self.all_cfgs.obs_dim))
        print("state_dim={}".format(self.all_cfgs.state_dim))
        print("action_dim={}".format(self.all_cfgs.action_dim))
        print("episode_limit={}".format(self.all_cfgs.episode_limit))

        # Create N agents
        try:
            self.agent_n = QMIX_SMAC(self.all_cfgs)
        except Exception as e:
            print(f'e:{e}')

        self.replay_buffer = ReplayBuffer(self.all_cfgs)

        self.epsilon = self.all_cfgs.epsilon  # Initialize the epsilon
        if self.all_cfgs.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=1)

        # output log dictionary
        self.log_dict_out = None
        print('done')

    def step(self, obs_dict, avail_a_n, *args, **kwargs):
        evaluate = kwargs['evaluate'] 

        #print(f'type:{type(obs_dict)} {type(avail_a_n)} {type(self.last_onehot_a_n)}')

        epsilon = 0 if evaluate else self.epsilon

        #print(f'obs_n:{obs_dict} last_onehot_a_n:{self.last_onehot_a_n} avail_a_n:{avail_a_n} epsilon:{epsilon}')        
        a_n = self.agent_n.choose_action(obs_dict, self.last_onehot_a_n, avail_a_n, epsilon)
        self.last_onehot_a_n = np.eye(self.all_cfgs.action_dim)[a_n]  # Convert actions to one-hot vectors
        return a_n
    
    # Add env_evaluation in the main trainable environemnt and the option for the evaluation

    def save(self, fname):
        pass

    def begin_episode(self, *args, **kwargs):
        if self.all_cfgs.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
            self.agent_n.eval_Q_net.rnn_hidden = None
        self.last_onehot_a_n = np.zeros((self.all_cfgs.N, self.all_cfgs.action_dim))  # Last actions of N agents(one-hot)
        self.log_dict_episode_out = dict()

    def end_episode(self, episode_step, obs_n, s, avail_a_n, *args, **kwargs):
        # An episode is over, store obs_n, s and avail_a_n in the last step
        self.replay_buffer.store_last_step(episode_step, obs_n, s, avail_a_n)

        if self.replay_buffer.current_size >= self.all_cfgs.batch_size:
            loss = self.agent_n.train(self.replay_buffer, episode_step)  # Training
            self.log_dict_episode_out['loss'] = loss

    def pre_episode_cycle(self, *args, **kwargs):
        self.log_dict_episode_cycle_out = dict()

    def post_episode_cycle(self, episode_step, obs_n, s, avail_a_n, agent_actions, rewards, dones, infos, *args, **kwargs):

        self.log_dict_episode_cycle_out['rewards'] = rewards

        if self.all_cfgs.use_reward_norm:
            rewards = self.reward_norm(rewards)
        """"
            When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
            dw means dead or win,there is no next state s';
            but when reaching the max_episode_steps,there is a next state s' actually.
        """
        if dones and episode_step + 1 != self.all_cfgs.episode_limit:
            dw = True
        else:
            dw = False

        # Store the transition
        #avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
        self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, self.last_onehot_a_n, agent_actions, rewards, dw)
        # Decay the epsilon
        self.epsilon = self.epsilon - self.all_cfgs.epsilon_decay if self.epsilon - self.all_cfgs.epsilon_decay > self.all_cfgs.epsilon_min else self.all_cfgs.epsilon_min

    def get(self, *args, **kwargs):
        what = kwargs['what']
        if what == 'log_dict_episode':
            return self.log_dict_episode_out
        elif what == 'log_dict_episode_cycle':
            return self.log_dict_episode_cycle_out
        return None
