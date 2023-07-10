import pygame
from pettingzoo.butterfly import knights_archers_zombies_v10

def main():
    print('ESC to exit')
    env = knights_archers_zombies_v10.env(render_mode="human")
    env.reset(seed=42)

    clock = pygame.time.Clock()
    manual_policy = knights_archers_zombies_v10.ManualPolicy(env)

    print(f'agents:{env.agents}')

    for agent in env.agent_iter():
        print(f'agent:{agent}')
        if env.render_mode == 'human':
            clock.tick(env.metadata["render_fps"])
        observation, reward, termination, truncation, info = env.last()
        
        if agent == manual_policy.agent:
            # get user input (controls are WASD and space)
            action = manual_policy(observation, agent)
        else:
            # this is where you would insert your policy (for non-player agents)
            action = env.action_space(agent).sample()

        # Check the content
        #print(f'action:{action}')
        #print(f'all:{observation, reward, termination, truncation, info}')
        env.step(action) 
    env.close()

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    main()