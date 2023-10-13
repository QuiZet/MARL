import numpy as np
from pettingzoo.mpe import simple_tag_v3

def random_action(action_space):
    """
    Generate a random action based on the action space.
    """
    return np.random.choice(action_space.n)

def main():
    # Initialize the AEC version of the environment to get agent names
    aec_env = simple_tag_v3.env()
    aec_env.reset()  # Reset the AEC environment
    agent_names = aec_env.agents

    # Initialize the parallel version of the environment for the rest of the code
    env = simple_tag_v3.parallel_env()

    # Lists to store observations and rewards
    observations_list = []
    rewards_list = []

    # Reset the environment to get initial observations
    observations = env.reset()
    print("Environment reset")

    # Number of steps to interact with the environment
    num_steps = 10  # Reduced for quicker debugging

    for _ in range(num_steps):
        print(f"Step {_ + 1}")
        actions = {}
        for agent in agent_names:
            action_space = env.action_space(agent)  # Use the action_space function
            actions[agent] = random_action(action_space)

        # Step the environment with the random actions
        next_observations, rewards, _, _, _ = env.step(actions)

        # Store the observations and rewards
        observations_list.append(next_observations)
        rewards_list.append(rewards)

        # Update the current observations
        observations = next_observations

    print("Observations:", observations_list)
    print("Rewards:", rewards_list)

if __name__ == "__main__":
    main()
