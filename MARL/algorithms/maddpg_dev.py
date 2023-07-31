import torch
import torch.nn.functional as F
from gymnasium.spaces import Box, Discrete
from MARL.utils.networks import MLPNetwork
from MARL.utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from MARL.utils.agents import DDPGAgent
import numpy as np

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0
        

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        agent_actions = []
        for i, agent in enumerate(self.agents):
            obs = observations[i]  # Observations for the current agent
            torch_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.pol_dev)
            action = agent.policy(torch_obs)

            if explore and self.is_training:
                action += agent.explore_noise.noise()

            agent_actions.append(action.squeeze(0).detach().cpu().numpy())

        return agent_actions

    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        # Convert observations, actions, and next_observations to tensors
        torch_obs = [torch.tensor(ob, dtype=torch.float32) for ob in obs]
        torch_next_obs = [torch.tensor(nob, dtype=torch.float32) for nob in next_obs]
        torch_actions = [torch.tensor(ac, dtype=torch.float32) for ac in acs]

        # Compute target values for the critic
        with torch.no_grad():
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies, torch_next_obs)]
            if self.alg_types[agent_i] == 'MADDPG':
                trgt_vf_in = torch.cat((*torch_next_obs, *all_trgt_acs), dim=1)
            else:
                trgt_vf_in = torch.cat((torch_next_obs[agent_i], all_trgt_acs[agent_i]), dim=1)
            target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                            curr_agent.target_critic(trgt_vf_in) *
                            (1 - dones[agent_i].view(-1, 1)))

        # Update critic
        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
        else:
            vf_in = torch.cat((torch_obs[agent_i], torch_actions[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value)
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        # Update policy
        curr_agent.policy_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = [pi(ob) for pi, ob in zip(self.policies, torch_obs)]
            vf_in = torch.cat((*torch_obs, *all_pol_acs), dim=1)
        else:
            vf_in = torch.cat((torch_obs[agent_i], curr_agent.policy(torch_obs[agent_i])), dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_agent.policy(torch_obs[agent_i])**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()

        if logger is not None:
            logger.add_scalars('agent%i/losses' % agent_i,
                               {'vf_loss': vf_loss,
                                'pol_loss': pol_loss},
                               self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        #env.agent_types]
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for atype in env.possible_agents]
        print(f'env.action_space:{env.action_space(env.possible_agents[0])}')
        print(f'env.observation_space:{env.observation_space(env.possible_agents[0])}')
        print(f'alg_types:{alg_types}')
        #for acsp, obsp, algtype in zip(env.action_space(env.possible_agents[0]), env.observation_space(env.possible_agents[0]),
        #                               alg_types):

        for agent in env.possible_agents:
            obsp = env.observation_space(agent)
            acsp = env.action_space(agent)
            algtype = alg_types[env.possible_agents.index(agent)]

            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)

            # Compute num_in_pol for each agent
            num_in_pol = obsp.shape[0]

            if algtype == "MADDPG":
                num_agents = 4
                num_in_critic = num_in_pol + num_out_pol * num_agents
            else:
                num_in_critic = num_in_pol + num_out_pol

            agent_init_params.append({
                'num_in_pol': num_in_pol,
                'num_out_pol': num_out_pol,
                'num_in_critic': num_in_critic
            })

        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                    'hidden_dim': hidden_dim,
                    'alg_types': alg_types,
                    'agent_init_params': agent_init_params,
                    'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance
        
    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
    