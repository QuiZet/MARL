import sys
sys.path.append('./')
sys.path.append('../MARL/')

import cv2
import torch 
import pygame
from pettingzoo.butterfly import cooperative_pong_v5

from MARL.algorithms.maddpg_naive import MADDPGAgent

def parallel_main():
    if True:
        print('Ctrl-Alt+C to exit')
        # https://pettingzoo.farama.org/api/parallel/
        parallel_env = cooperative_pong_v5.parallel_env(render_mode='human')
        obs = parallel_env.reset()

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create MADDPG agents
        # Get the observation and action spaces
        obs_space = parallel_env.observation_spaces["paddle_0"].shape[0] * parallel_env.observation_spaces["paddle_0"].shape[1] * parallel_env.observation_spaces["paddle_0"].shape[2]
        # MADDPAgent takes the action space size
        # Discrete action space may differ from continuous
        # https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete
        act_space = 1 #parallel_env.action_spaces["paddle_0"].n
        num_agents = len(parallel_env.possible_agents)
        
        # Create MADDPG agents
        agents = [MADDPGAgent(device, num_agents, obs_space, act_space) for _ in range(num_agents)]

        episode_rewards = []

        clock = pygame.time.Clock()

        num_episodes=1000
        max_steps=100

        for episode in range(num_episodes):
            obs = parallel_env.reset()
            #print(f'obs:{obs}')
            obs = obs[0]
            #print(f'obs:{obs}')
            agent_names = list(parallel_env.agents)

            total_reward = 0

            for step in range(max_steps):
                clock.tick(parallel_env.metadata["render_fps"])
                actions_for_env = dict()

                # Select actions for each agent
                for agent_idx in range(num_agents):
                    agent = agents[agent_idx]
                    if obs is not None and obs[agent_names[agent_idx]] is not None:
                        action = agent.select_action(torch.flatten(torch.Tensor(obs[agent_names[agent_idx]])).to(device))
                        actions_for_env[agent_names[agent_idx]] = int(action[0])
                        if actions_for_env[agent_names[agent_idx]] > 1: actions_for_env[agent_names[agent_idx]] = 1
                        if actions_for_env[agent_names[agent_idx]] < -1: actions_for_env[agent_names[agent_idx]] = -1

                if len(actions_for_env) == 0:
                    break

                #print(f'actions:{actions_for_env}')
                next_obs, rewards, terminations, truncations, infos = parallel_env.step(actions_for_env)
                #print(f'all:{next_obs, rewards, terminations, truncations, infos}')

                # Update each agent
                for agent_idx in range(num_agents):
                    agent = agents[agent_idx]
                    agent.update(
                        obs[agent_names[agent_idx]] if obs is not None else None,
                        actions_for_env[agent_names[agent_idx]],
                        {agent_names[agent_idx]: rewards[agent_names[agent_idx]]},
                        next_obs[agent_names[agent_idx]]
                    )

                # Print rewards
                print("Rewards:")
                for agent_name, reward in rewards.items():
                    print(f"Agent {agent_name}: {reward}")


                obs = next_obs
                total_reward += sum(rewards.values())

                cv2.imshow('observation', next_obs['paddle_1'])
                cv2.waitKey(1)

                # Check the content
                if True in terminations:
                    break
                if True in truncations:
                    break
                    #parallel_env.reset()

                input()

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

            # Print selected actions
            print("Selected Actions:")
            for agent_idx, action in enumerate(actions_for_env):
                print(f"Agent {agent_idx}: {action}")


if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    parallel_main()