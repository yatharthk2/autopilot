import gym
import highway_env
from matplotlib import pyplot as plt
env = gym.make("roundabout-v0")
env.reset()
for _ in range(300):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
plt.imshow(env.render(mode="rgb_array"))
plt.show()
