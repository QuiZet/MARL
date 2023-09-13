from pettingzoo.mpe import simple_tag_v3

env = simple_tag_v3.env()

env.reset()
print(env.state_space)