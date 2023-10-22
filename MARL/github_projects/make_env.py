import sys
print(f'sys_path: {sys.path}')

def make_env(scenario_name, benchmark=False, discrete=False):
    from environment import MultiAgentEnv
    import scenarios_pkg as scenarios_pkg
    #from scenarios_pkg import scenario_name
    
    # import all scenarios as modules
    directory_path = r"/home/yungisimon/MARL/MARL/github_projects/scenarios_pkg"
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