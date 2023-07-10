import pygame
from pettingzoo.butterfly import cooperative_pong_v5

import cv2

def main():
    print('Ctrl-Alt+C to exit')
    env = cooperative_pong_v5.env(render_mode="human")
    env.reset(seed=42)

    clock = pygame.time.Clock()
    manual_policy = cooperative_pong_v5.ManualPolicy(env)

    num_reset = 0
    for agent in env.agent_iter():
        print(f'agent:{agent}')
        if env.render_mode == 'human':
            clock.tick(env.metadata["render_fps"])
        observation, reward, termination, truncation, info = env.last()

        if env.render_mode == 'human':
            cv2.imshow('observation', observation)
            cv2.waitKey(1)
        
        if agent == manual_policy.agent:
            # get user input (controls are WASD and space)
            action = manual_policy(observation, agent)
        else:
            # this is where you would insert your policy (for non-player agents)
            action = env.action_space(agent).sample()

        # Check the content
        print(f'action:{action}')
        print(f'all:{observation, reward, termination, truncation, info}')
        if termination or truncation:
            env.reset()
            num_reset += 1
        else:
            env.step(action) 
        if num_reset > 1: break
    env.close()

def parallel_main():
    try:
        print('Ctrl-Alt+C to exit')
        # https://pettingzoo.farama.org/api/parallel/
        parallel_env = cooperative_pong_v5.parallel_env(render_mode='human')
        observations = parallel_env.reset()

        clock = pygame.time.Clock()

        num_reset = 0
        while parallel_env.agents:
            clock.tick(parallel_env.metadata["render_fps"])
            # this is where you would insert your policy
            actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  
            print(f'actions:{actions}')
            
            observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
            print(f'all:{observations, rewards, terminations, truncations, infos}')

            cv2.imshow('observation', observations['paddle_1'])
            cv2.waitKey(1)

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