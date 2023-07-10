import sys
sys.path.append('./')
sys.path.append('../MARL/')

import numpy as np
import cv2
import torch 
import pygame
from pettingzoo.mpe import simple_tag_v3

from MARL.algorithms.maddpg_naive import MADDPGAgent

def parallel_main():
    if True:
        print('Ctrl-Alt+C to exit')
        # https://pettingzoo.farama.org/api/parallel/
        parallel_env = simple_tag_v3.parallel_env(render_mode='human', num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=True)
        obs = parallel_env.reset()

        # Set the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create MADDPG agents
        num_agents = len(parallel_env.possible_agents)

        agents = []        
        for agent in parallel_env.possible_agents:
            obs_space = parallel_env.observation_spaces[agent].shape[0]
            act_space = 5#parallel_env.action_spaces[agent].n
            maddpg_agent = MADDPGAgent(device, num_agents, obs_space, act_space)
            agents.append(maddpg_agent)

        episode_rewards = []

        clock = pygame.time.Clock()

        num_episodes=1000
        max_steps=100
        agent_names = list(parallel_env.agents)

        for episode in range(num_episodes):
            obs = parallel_env.reset()
            #print(f'obs:{obs}')
            obs = obs[0]
            #print(f'obs:{obs}')

            total_reward = 0

            for step in range(max_steps):
                clock.tick(parallel_env.metadata["render_fps"])
                actions_for_env = dict()

                # Select actions for each agent
                for agent_idx in range(num_agents):
                    agent = agents[agent_idx]

                    #print(f'agent_names:{agent_names} {agent_idx}')
                    #print(f'obs:{obs}')

                    if obs is not None and obs[agent_names[agent_idx]] is not None:
                        action = agent.select_action(torch.flatten(torch.Tensor(obs[agent_names[agent_idx]])).to(device))
                        #print(f'action action:{action}')
                        actions_for_env[agent_names[agent_idx]] = np.array(action, dtype=np.float32)

                #print(f'actions_for_env:{type(actions_for_env)} {actions_for_env}')

                if len(actions_for_env) == 0:
                    break

                #print(f'actions:{actions_for_env}')
                next_obs, rewards, terminations, truncations, infos = parallel_env.step(actions_for_env)
                #print(f'all:{next_obs, rewards, terminations, truncations, infos}')
                # It seems an error in the pipeline
                if len(next_obs) == 0:
                    #print('len is 0')
                    break

                # Update each agent (some agent may not exist because not visible)
                # The exception should manage the case (not best solution, it does not capture network errors).
                critic_losses = []
                actor_losses = []
                for agent_idx in range(num_agents):
                    try:
                        agent = agents[agent_idx]
                        critic_loss, actor_loss = agent.update(
                            obs[agent_names[agent_idx]] if obs is not None else None,
                            actions_for_env[agent_names[agent_idx]],
                            {agent_names[agent_idx]: rewards[agent_names[agent_idx]]},
                            next_obs[agent_names[agent_idx]]
                        )
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                    except Exception:
                        pass

                # Print rewards
                print(f'rewards:{rewards}')
                if step % 20 == 0:
                    print("Rewards:")
                    for agent_name, reward in rewards.items():
                        print(f"Agent {agent_name}: {reward}")
                    print(f"losses critic_losses:{critic_losses}")
                    print(f"losses actor_losses:{actor_losses}")


                obs = next_obs
                total_reward += sum(rewards.values())

                # Check the content
                if True in terminations:
                    break
                if True in truncations:
                    #parallel_env.reset()
                    break

            episode_rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

            # Print selected actions
            #print("Selected Actions:")
            #for agent_idx, action in enumerate(actions_for_env):
            #    print(f"Agent {agent_idx}: {action}")


if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    parallel_main()