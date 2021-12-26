import gym 
env = gym.make('highway-fast-v0').unwrapped
env.config['lanes_count'] = 3