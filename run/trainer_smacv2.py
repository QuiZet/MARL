import time
import os
import importlib

from omegaconf import OmegaConf
import numpy as np

from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from MARL.utils.dictionary import AttrDict
from MARL.algorithms.qmix.replay_buffer import ReplayBuffer
from MARL.algorithms.qmix.qmix_smac import QMIX_SMAC
from MARL.algorithms.qmix.normalization import Normalization

def run_parallel_smacv2(env, env_evaluate, model, logger, env_config, model_config, *args, **kwargs):
    distribution_config = {
        "n_units": 3,
        "n_enemies": 3,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [1.0, 0, 0],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "map_x": 32,
            "map_y": 32,
        },
    }


    env = StarCraftCapabilityEnvWrapper(
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=False,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
    )
    env_eval = env

    print(f'model_config:{model_config}')
    print(f'env_config:{env_config}')

    # Create env
    env_info = env.get_env_info()
    model_config['N'] = env_info["n_agents"]  # The number of agents
    model_config['obs_dim'] = env_info["obs_shape"]  # The dimensions of an agent's observation space
    model_config['state_dim'] = env_info["state_shape"]  # The dimensions of global state space
    model_config['action_dim'] = env_info["n_actions"]  # The dimensions of an agent's action space
    model_config['episode_limit'] = env_info["episode_limit"]  # Maximum number of steps per episode
    model_config['epsilon_decay'] = (model_config['epsilon'] - model_config['epsilon_min']) / model_config['epsilon_decay_steps']

    m_config = OmegaConf.to_container(model_config)
    e_config = OmegaConf.to_container(env_config)
    args = m_config | e_config
    args = AttrDict(args)
    print(f'args:{args}')

    print("number of agents={}".format(args.N))
    print("obs_dim={}".format(args.obs_dim))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("episode_limit={}".format(args.episode_limit))

    # Create N agents
    agent_n = QMIX_SMAC(args)
    replay_buffer = ReplayBuffer(args)

    run(args, env, agent_n, replay_buffer)

def run(args, env, agent_n, replay_buffer):

    win_rates = []  # Record the win rates
    total_steps = 0
    epsilon = args.epsilon  # Initialize the epsilon

    reward_norm = None
    if args.use_reward_norm:
        print("------use reward norm------")
        reward_norm = Normalization(shape=1)


    evaluate_num = -1  # Record the number of evaluations
    while total_steps < args.max_train_steps:
        if total_steps // args.evaluate_freq > evaluate_num:
            win_rates, total_steps, epsilon = evaluate_policy(env, args, agent_n, replay_buffer, reward_norm, epsilon, win_rates, total_steps)  # Evaluate the policy every 'evaluate_freq' steps
            evaluate_num += 1

        _, _, episode_steps, epsilon = run_episode_smac(env, args, agent_n, replay_buffer, reward_norm, epsilon, evaluate=False)  # Run an episode
        total_steps += episode_steps

        if replay_buffer.current_size >= args.batch_size:
            agent_n.train(replay_buffer, total_steps)  # Training

    win_rates, total_steps, epsilon = evaluate_policy(env, args, agent_n, replay_buffer, reward_norm, epsilon, win_rates, total_steps)
    env.close()

def evaluate_policy(env, args, agent_n, replay_buffer, reward_norm, epsilon, win_rates, total_steps):
    win_times = 0
    evaluate_reward = 0
    for _ in range(args.evaluate_times):
        win_tag, episode_reward, _, epsilon = run_episode_smac(env, args, agent_n, replay_buffer, reward_norm, epsilon, evaluate=True)
        if win_tag:
            win_times += 1
        evaluate_reward += episode_reward

    win_rate = win_times / args.evaluate_times
    evaluate_reward = evaluate_reward / args.evaluate_times
    win_rates.append(win_rate)
    print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(total_steps, win_rate, evaluate_reward))
    return win_rates, total_steps, epsilon

def run_episode_smac(env, args, agent_n, replay_buffer, reward_norm, epsilon, evaluate=False):
    win_tag = False
    episode_reward = 0
    env.reset()
    if args.use_rnn:  # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
        agent_n.eval_Q_net.rnn_hidden = None
    last_onehot_a_n = np.zeros((args.N, args.action_dim))  # Last actions of N agents(one-hot)
    for episode_step in range(args.episode_limit):
        obs_n = env.get_obs()  # obs_n.shape=(N,obs_dim)
        s = env.get_state()  # s.shape=(state_dim,)
        avail_a_n = env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
        epsilon = 0 if evaluate else epsilon
        a_n = agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)
        last_onehot_a_n = np.eye(args.action_dim)[a_n]  # Convert actions to one-hot vectors
        r, done, info = env.step(a_n)  # Take a step
        win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
        episode_reward += r

        env.render()

        if not evaluate:
            if reward_norm:
                r = reward_norm(r)
            """"
                When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                dw means dead or win,there is no next state s';
                but when reaching the max_episode_steps,there is a next state s' actually.
            """
            if done and episode_step + 1 != args.episode_limit:
                dw = True
            else:
                dw = False

            # Store the transition
            replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
            # Decay the epsilon
            epsilon = epsilon - args.epsilon_decay if epsilon - args.epsilon_decay > args.epsilon_min else args.epsilon_min

        if done:
            break

    if not evaluate:
        # An episode is over, store obs_n, s and avail_a_n in the last step
        obs_n = env.get_obs()
        s = env.get_state()
        avail_a_n = env.get_avail_actions()
        replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n)

    return win_tag, episode_reward, episode_step + 1, epsilon