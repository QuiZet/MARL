import random
from collections import deque
import numpy as np

import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from MARL.utils.dictionary import AttrDict
from MARL.models.abstractwrapper import AbstractWrapper




class ReplayBuffer():
    def __init__(self, device, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.tensor(a_lst, dtype=torch.float).to(self.device), \
                torch.tensor(r_lst, dtype=torch.float).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                torch.tensor(done_mask_lst, dtype=torch.float).to(self.device)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate, input_size, output_size, init_alpha, lr_alpha, target_entropy):
        super(PolicyNet, self).__init__()
        self.target_entropy = target_entropy
        self.fc1 = nn.Linear(input_size, 128)
        self.fc_mu = nn.Linear(128,output_size)
        self.fc_std  = nn.Linear(128,output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate, input_size, output_size, tau):
        super(QNet, self).__init__()
        self.tau = tau
        self.fc_s = nn.Linear(input_size, 64)
        self.fc_a = nn.Linear(output_size,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        s, a, r, s_prime, done = mini_batch
        loss = F.smooth_l1_loss(self.forward(s, a) , target)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target):
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
    
class Agent:
    def __init__(self, device, buffer_limit, batch_size, state_dim, action_dim, lr_q, lr_pi, init_alpha, lr_alpha, target_entropy, tau, gamma):
        self.batch_size = batch_size
        self.memory = ReplayBuffer(device, buffer_limit=buffer_limit)
        self.q1 = QNet(lr_q, state_dim, action_dim, tau).to(device)
        self.q2 = QNet(lr_q, state_dim, action_dim, tau).to(device)
        self.q1_target = QNet(lr_q, state_dim, action_dim, tau).to(device)
        self.q2_target = QNet(lr_q, state_dim, action_dim, tau).to(device)
        self.pi = PolicyNet(lr_pi, state_dim, action_dim, init_alpha, lr_alpha, target_entropy).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.gamma = gamma

    def act(self, state):
        return self.pi(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))

    def replay(self):
        if self.memory.size()>1000:
            for i in range(20):
                mini_batch = self.memory.sample(self.batch_size)
                td_target = self.calc_target(self.pi, self.q1_target, self.q2_target, mini_batch, self.gamma)
                self.q1.train_net(td_target, mini_batch)
                self.q2.train_net(td_target, mini_batch)
                entropy = self.pi.train_net(self.q1, self.q2, mini_batch)
                self.q1.soft_update(self.q1_target)
                self.q2.soft_update(self.q2_target)

    def calc_target(self, pi, q1, q2, mini_batch, gamma):
        s, a, r, s_prime, done = mini_batch

        with torch.no_grad():
            a_prime, log_prob= pi(s_prime)
            entropy = -pi.log_alpha.exp() * log_prob
            q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
            q1_q2 = torch.cat([q1_val, q2_val], dim=1)
            min_q = torch.min(q1_q2, 1, keepdim=True)[0]
            target = r + gamma * done * (min_q + entropy)

        return target

class MinimalWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')

        #Hyperparameters
        lr_pi           = 0.0005
        lr_q            = 0.001
        init_alpha      = 0.01
        gamma           = 0.98
        batch_size      = 32
        buffer_limit    = 50000
        tau             = 0.01 # for target network soft update
        target_entropy  = -1.0 # for automated alpha update
        lr_alpha        = 0.001  # for automated alpha update

        self.agents = dict()
        for agent in self.env.possible_agents:

            print(f'agent:{agent}')
            state_dim = self.env.observation_space(agent).shape[0]
            if isinstance(self.env.action_space(agent), gym.spaces.Box):
                action_dim = self.env.action_space(agent).shape[0] 
            else:
                action_dim = self.env.action_space(agent).n
            agent_obj = Agent(device=self.device, buffer_limit=buffer_limit, batch_size=batch_size, state_dim=state_dim, action_dim=action_dim, lr_q=lr_q, lr_pi=lr_pi, init_alpha=init_alpha, lr_alpha=lr_alpha, target_entropy=target_entropy, tau=tau, gamma=gamma)
            self.agents[agent] = agent_obj

    def step(self, obs_dict, *args, **kwargs):
        print('============================================')

        # Convert the observations to torch Tensor
        torch_obs = []
        for agent in self.env.possible_agents:
            agent_obs = [obs_dict[agent]]
            torch_obs.append(Variable(torch.Tensor(agent_obs), requires_grad=False))

        agent_actions = dict()
        for i, agent in enumerate(self.env.possible_agents):
            #print(f'i:{i} agent:{agent} torch_obs:{torch_obs}')
            action = self.agents[agent].act(torch_obs[i].to(self.device))  # this is where you would insert your policy
            agent_actions[agent] = action[0].squeeze(0).cpu().detach().numpy()#[0.1, 0.0, 0.01, 0.0, 0.01] #action.squeeze(0).cpu().detach().numpy()
        print(f'agent_actions:{agent_actions}')
        return agent_actions

    def save(self, fname):
        pass

    def begin_episode(self, ep_i, *args, **kwargs):
        pass

    def end_episode(self, *args, **kwargs):
        pass

    def pre_episode_cycle(self, *args, **kwargs):
        pass

    def post_episode_cycle(self, *args, **kwargs):
        ep_cycle_i, obs_dict, agent_actions, rewards, next_obs, dones = args
        # (array([ 0.9309764, -0.3650794,  0.3355128], dtype=float32), tensor([0.0765], device='cuda:0', grad_fn=<TanhBackward0>), -0.15094553303113115, array([ 0.9325135 , -0.3611352 ,  0.08466274], dtype=float32), False)

        #print(f'post: obs_dict{obs_dict} agent_actions:{agent_actions} rewards:{rewards} next_obs:{next_obs} dones:{dones}')
        for agent in self.env.possible_agents:
            self.agents[agent].remember(
                obs_dict[agent], agent_actions[agent], rewards[agent], next_obs[agent], dones[agent])
            self.agents[agent].replay()
