import gym 
import highway_env
env = gym.make('highway-v0')
env.seed(0)
# print('State shape: ', env.observation_space.shape)
# print('Number of actions: ', env.action_space.n)