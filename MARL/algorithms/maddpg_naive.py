import torch
import torch.nn as nn
import torch.optim as optim

# Define the actor network
class ActorNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space, 64)
        self.fc2 = nn.Linear(64, action_space)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

# Define the critic network
class CriticNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_space + action_space, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, obs, actions):
        x = torch.cat((obs, actions), dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# %%
# Define the MADDPG agent
class MADDPGAgent:
    def __init__(self, device, num_agents, observation_space, action_space):
        print(f'MADDPAgent::__init__ device, num_agents, observation_space, action_space:{device, num_agents, observation_space, action_space}')
        self.device = device
        self.num_agents = num_agents
        self.actor_network = ActorNetwork(observation_space, action_space).to(device)
        self.critic_network = CriticNetwork(observation_space, action_space).to(device)
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
    
    def select_action(self, obs):
        obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.actor_network(obs).cpu().numpy().squeeze(0).tolist()  # Convert to list
        #print(f'MADDPAgent::select_action obs, actions:{obs, actions}')
        return actions

    
    def update(self, obs, actions, rewards, next_obs):
        #print(f'MADDPAgent::update obs, actions, rewards, next_obs:{obs, actions, rewards, next_obs}')
        obs = torch.flatten(torch.Tensor(obs)).to(self.device)
        actions = torch.flatten(torch.Tensor(actions).unsqueeze(0)).to(self.device)
        rewards = torch.flatten(torch.Tensor(list(rewards.values())).unsqueeze(0)).to(self.device)  # Convert rewards to a list
        next_obs = torch.flatten(torch.Tensor(next_obs)).to(self.device)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        q_values = self.critic_network(obs, actions)
        #print(f'MADDPAgent::update obs, actions, q_values, rewards:{obs, actions, q_values, rewards}')

        next_actions = self.actor_network(next_obs)
        next_q_values = self.critic_network(next_obs, next_actions)
        discount_factor = 0.99
        td_targets = rewards + discount_factor * next_q_values

        critic_loss = nn.MSELoss()(q_values, td_targets)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actor_actions = self.actor_network(obs)
        actor_loss = -self.critic_network(obs, actor_actions.detach()).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss, actor_loss

        if False:
            # Print the gradients
            actor_grads = [param.grad for param in self.actor_network.parameters()]
            critic_grads = [param.grad for param in self.critic_network.parameters()]

            #print("Actor Gradients:")
            for grad in actor_grads:
                print(grad)

            #print("Critic Gradients:")
            for grad in critic_grads:
                print(grad)
