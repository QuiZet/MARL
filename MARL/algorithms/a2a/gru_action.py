
import numpy as np
import torch
import torch.nn as nn
from pettingzoo.mpe import simple_tag_v3
import gym

class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out)
        return self.softmax(out).squeeze(1)

def test_model(model, env, agent_names, output_dim, num_test_steps=10):
    model.eval() 
    total_rewards = {agent: 0 for agent in agent_names}
    
    with torch.no_grad():
        observations, _ = env.reset()
        for _ in range(num_test_steps):
            actions = {agent: np.zeros(output_dim) for agent in agent_names}
            rewards = {agent: 0 for agent in agent_names}
            obs_rewards = []
            for agent in agent_names:
                obs_reward = np.append(observations[agent], rewards[agent])
                if agent == "agent_0":
                    obs_reward = np.append(obs_reward, [0, 0])
                obs_rewards.append(obs_reward)

            obs_rewards_np = np.array(obs_rewards)
            obs_rewards = torch.tensor(obs_rewards_np, dtype=torch.float32)
            action_probs = model(obs_rewards)
            for idx, agent in enumerate(agent_names):
                chosen_action = np.random.choice(output_dim, p=action_probs[idx].numpy())
                actions[agent][chosen_action] = 1
            
            next_observations, reward_dicts, _, _, _ = env.step(actions)
            for agent in agent_names:
                rewards[agent] += reward_dicts[agent]
            
            observations = next_observations

    for agent in agent_names:
        print(f"Total reward for {agent}: {total_rewards[agent]}")

def main():
    # init environment
    aec_env = simple_tag_v3.env()
    aec_env.reset()
    agent_names = aec_env.agents
    env = simple_tag_v3.parallel_env(continuous_actions=True)

    # GRU network
    input_dim = env.observation_space(agent_names[0]).shape[0] + 1  # observation + reward
    hidden_dim = 32
    output_dim = 5  # [no_action, move_left, move_right, move_down, move_up]
    model = GRUNetwork(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # reset env
    observations, info = env.reset()

    num_steps = 10
    print(f'agent_names:{agent_names}')
    
    for _ in range(num_steps):
        actions = {agent: [0.00,0.00,0.00,0.00,0.00] for agent in agent_names}
        rewards = {agent: 0 for agent in agent_names}
        obs_rewards = []
        print(f'print actions that will be taken:{actions}')
        next_observations, reward_dicts, _, _, _ = env.step(actions)
        for agent in agent_names:
            rewards[agent] = reward_dicts[agent]

        print(f'type(observations):{type(observations)}')
        print(f'observations:{observations}')
        for agent in agent_names:
            obs_reward = np.append(observations[agent], rewards[agent])
            # zero pad the agent observation with 0 to match adversary
            if agent == "agent_0":
                obs_reward = np.append(obs_reward, [0, 0])
            obs_rewards.append(obs_reward)

        # Convert the list of numpy arrays to a single numpy array
        obs_rewards_np = np.array(obs_rewards)
        # Convert the numpy array to a tensor
        obs_rewards = torch.tensor(obs_rewards_np, dtype=torch.float32)
        # get action probs from the GRU
        action_probs = model(obs_rewards)
        # choose actions based on the probs
        for idx, agent in enumerate(agent_names):
            actions[agent] = np.random.choice(output_dim, p=action_probs[idx].detach().numpy())

        # calc loss
        targets = torch.tensor(list(actions.values()), dtype=torch.long)
        loss = criterion(action_probs, targets)

        # Update the current observations
        observations = next_observations

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("-----Training completed-----")
        save_path = './models/gru_action.pt'
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # test model
        test_model(model, env, agent_names, output_dim)

if __name__ == "__main__":
    
    main()

