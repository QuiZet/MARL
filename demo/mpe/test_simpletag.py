import pygame
from pettingzoo.mpe import simple_tag_v3

def main():
    try:
        print('Ctrl-Alt+C to exit')

        env = simple_tag_v3.env(render_mode='human')
        env.reset()

        clock = pygame.time.Clock()

        num_reset = 0
        for agent in env.agent_iter():
            print(f'agent:{agent}')
            clock.tick(env.metadata["render_fps"])
            observation, reward, termination, truncation, info = env.last()
            
            # this is where you would insert your policy (for non-player agents)
            action = env.action_space(agent).sample()

            # Check the content
            #print(f'action:{action}')
            #print(f'all:{observation, reward, termination, truncation, info}')
            if termination or truncation:
                env.reset()
                num_reset += 1
            else:
                env.step(action) 
            if num_reset > 1: break
        env.close()

    except Exception as e:
        print(f'ex:{e}')

def parallel_main():
    try:
        print('Ctrl-Alt+C to exit')
        # https://pettingzoo.farama.org/api/parallel/
        parallel_env = simple_tag_v3.parallel_env(render_mode='human', continuous_actions=True)
        observations = parallel_env.reset()

        clock = pygame.time.Clock()

        num_reset = 0
        while parallel_env.agents:
            clock.tick(parallel_env.metadata["render_fps"])
            # this is where you would insert your policy
            actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  
            print(f'actions:{type(actions)} {actions}')
            
            observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
            print(f'all:{observations, rewards, terminations, truncations, infos}')

            # Check the content
            if True in terminations or True in truncations:
                parallel_env.reset()
                num_reset += 1
            if num_reset > 1: break
        parallel_env.close()

    except Exception as e:
        print(f'ex:{e}')

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    #main()
    parallel_main()