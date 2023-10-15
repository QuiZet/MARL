from pettingzoo.mpe import simple_tag_v3
import numpy as np

def main():
    # Initialize the environment
    env = simple_tag_v3.parallel_env(continuous_actions=True)
    env.reset()

    num_steps = 10

    for _ in range(num_steps):
        actions = {}
        for agent in env.agents:
            # Sample a random action for the agent
            action_space = env.action_space(agent)
            random_action = action_space.sample()
            actions[agent] = random_action

        # Pass the actions to the environment
        next_observations, reward_dicts, _, _, _ = env.step(actions)
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")

    env.close()

if __name__ == "__main__":
    main()
