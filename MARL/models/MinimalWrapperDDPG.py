import random
from collections import deque
import numpy as np

from gymnasium.spaces import Box

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from MARL.utils.dictionary import AttrDict
from MARL.models.abstractwrapper import AbstractWrapper


class DDPG(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DDPG, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Agent:
    def __init__(self, device, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.actor_model = DDPG(state_dim, action_dim).to(device)
        self.critic_model = DDPG(state_dim + action_dim, 1).to(device)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.learning_rate_critic)

    def act(self, state):
        return self.actor_model(state)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)

        # cat -> stack
        states = torch.stack([s for (s,a,r,n,d) in batch])
        actions = torch.stack([a for (s,a,r,n,d) in batch])
        rewards = torch.stack([r for (s,a,r,n,d) in batch]).float()
        next_states = torch.stack([n for (s,a,r,n,d) in batch])
        dones = torch.stack([d for (s,a,r,n,d) in batch]).float()

        q_values_next_state = self.critic_model(torch.cat((next_states.detach(),self.actor_model(next_states.detach())),dim=1)).detach()
        
        q_values_target = rewards + (self.gamma * q_values_next_state * (1 - dones))
        
        q_values_current_state = self.critic_model(torch.cat((states.detach(),actions.detach()),dim=1))

        print(f'q_values_current_state:{q_values_current_state} q_values_next_state:{q_values_next_state}')

        critic_loss = F.mse_loss(q_values_current_state, q_values_target)

        actor_loss = -self.critic_model(torch.cat((states,self.actor_model(states)),dim=1)).mean()

        self.actor_optimizer.zero_grad()
        
        actor_loss.backward()
        
        self.actor_optimizer.step()

        
        self.critic_optimizer.zero_grad()
        
        critic_loss.backward()
        
        self.critic_optimizer.step()

        print(f'loss actor_loss:{actor_loss} critic_loss:{critic_loss}')

    def decay_epsilon(self):
        
        self.epsilon *= self.epsilon_decay



class MinimalWrapper(AbstractWrapper):
    def __init__(self, *args, **kwargs):
        print(f"args:{args}")
        print(f'kwargs:{kwargs}')

        self.config = AttrDict(kwargs)
        print(f'self.config:{self.config}')

        self.env = args[0]
        self.device = args[1]
        print(f'device:{self.device}')
        self.t = 0

        self.agents = dict()
        for agent in self.env.possible_agents:

            print(f'agent:{agent}')
            state_dim = self.env.observation_space(agent).shape[0]
            if isinstance(self.env.action_space(agent), Box):
                action_dim = self.env.action_space(agent).shape[0] 
            else:
                action_dim = self.env.action_space(agent).n
            agent_obj = Agent(device=self.device, state_dim=state_dim, action_dim=action_dim)
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
            agent_actions[agent] = action.squeeze(0).cpu().detach().numpy()#[0.1, 0.0, 0.01, 0.0, 0.01] #action.squeeze(0).cpu().detach().numpy()
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
        print(f'post: obs_dict{obs_dict} agent_actions:{agent_actions} rewards:{rewards} next_obs:{next_obs} dones:{dones}')
        for agent in self.env.possible_agents:
            self.agents[agent].remember(
                torch.tensor(obs_dict[agent]).to(self.device), 
                torch.tensor(agent_actions[agent]).to(self.device), 
                torch.tensor(np.array([rewards[agent]])).to(self.device), 
                torch.tensor(next_obs[agent]).to(self.device), 
                torch.tensor(np.array([dones[agent]])).to(self.device))
            self.agents[agent].replay()
