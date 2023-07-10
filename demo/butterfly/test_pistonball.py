from pettingzoo.butterfly import pistonball_v6

def main():
    print('Ctrl-Alt+C to exit')

    env = pistonball_v6.parallel_env(render_mode="human")
    observations = env.reset()

    while env.agents:
        # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}  

        print(f'agents:{env.agents}')
        print(f'actions:{actions}')

        observations, rewards, terminations, truncations, infos = env.step(actions)
    env.close()

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    main()