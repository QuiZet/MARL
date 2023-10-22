import sys
print(f'sys_path: {sys.path}')

def make_env(scenario_name, benchmark=False, discrete=False):
    from environment import MultiAgentEnv
    import scenarios_pkg as scenarios_pkg

    # import all scenarios as modules
    scenario = scenarios_pkg.load
    print(scenario_name)
    # create world
    world = scenario_name.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data, discrete=discrete)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, discrete=discrete)
    return env