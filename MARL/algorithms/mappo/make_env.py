import sys
from pettingzoo.mpe import simple_adversary_v3, simple_crypto_v3, simple_push_v3, simple_reference_v3, simple_speaker_listener_v4, simple_spread_v3, simple_tag_v3, simple_world_comm_v3

print(f'sys_path: {sys.path}')

def make_env(scenario_name, benchmark=False, discrete=False):
    from environment import MultiAgentEnv
    import scenarios_pkg as scenarios_pkg
    #from scenarios_pkg import scenario_name
    
    # import all scenarios as modules
    directory_path = 'path_to_scenarios_pkg'
    loaded_modules = scenarios_pkg.load(directory_path)
    print(f'loaded_modules: {loaded_modules}')
    scenario = loaded_modules[scenario_name].Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, discrete=discrete)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete=discrete)
    return env

env_mapping = {
    "simple":simple_adversary_v3.parallel_env,
    "simple_adversary": simple_adversary_v3.parallel_env,
    "simple_crypto": simple_crypto_v3.parallel_env,
    "simple_push": simple_push_v3.parallel_env,
    "simple_reference": simple_reference_v3.parallel_env,
    "simple_speaker_listener": simple_speaker_listener_v4.parallel_env,
    "simple_spread": simple_spread_v3.parallel_env,
    "simple_tag": simple_tag_v3.parallel_env,
    "simple_world_comm": simple_world_comm_v3.parallel_env,
}

def make_env_from_pettingzoo(env_name, **kwargs):
    if env_name in env_mapping:
        return env_mapping[env_name](**kwargs)
    else:
        raise ValueError(f"Unrecognized environment name {env_name}")