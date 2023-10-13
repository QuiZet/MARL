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

def main():
    # init environment
    aec_env = simple_tag_v3.env()
    aec_env.reset()
    agent_names = aec_env.agents
    env = simple_tag_v3.parallel_env(continuous_actions=False)

    # GRU network
    input_dim = env.observation_space(agent_names[0]).shape[0] + 1  # observation + reward
    hidden_dim = 32
    output_dim = 5  # [no_action, move_left, move_right, move_down, move_up]
    model = GRUNetwork(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # reset env
    observations = env.reset()

    num_steps = 10

    for _ in range(num_steps):
        actions = {}
        rewards = {agent: 0 for agent in agent_names}
        obs_rewards = []

        next_observations, reward_dicts, _, _, _ = env.step(actions)
        for agent in agent_names:
            rewards[agent] = reward_dicts[agent].get('reward', 0.0)

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

if __name__ == "__main__":
    main()
