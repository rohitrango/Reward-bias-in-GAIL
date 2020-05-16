import gym_minigrid, gym
from gym_minigrid import wrappers as mgwr
import numpy as np

keep_classes = ['goal', 'agent', 'wall', 'empty']
drop_color =1
env = gym.make('MiniGrid-DoorKey-6x6-v0')
print("env obs space: ", env.observation_space)
# print(env.observation_space)
keep_classes.extend(['door', 'key'])

print("state is: ", env.observation)
env = mgwr.FullyObsWrapper(env)
env = mgwr.ImgObsWrapper(env)
env = mgwr.FullyObsOneHotWrapper(env, drop_color=drop_color, keep_classes=keep_classes, flatten=False)

# print("steps: ", env.max_steps)
# print("after wrapping action space: ", env.action_space)
# print("after wrapping action space shape is: ", env.action_space.shape)
# print("after wrapping obs space: ", env.observation_space)
# print("shape of obs space: ", env.observation_space.shape)

# print("after wrapping obs space low is: ", env.observation_space.low.shape)
# print("after wrapping obs space high is: ", env.observation_space.high.shape)

obs_space = env.observation_space
# print(obs_space.shape[2])
# print((obs_space.shape[0], obs_space[1], obs_space[2]))
observation_space = gym.spaces.Box(shape=((obs_space.shape[0], obs_space.shape[1], obs_space.shape[2])),
                					low=0,
                					high=255)


# print("after wrapping obs space: ", env.observation_space)
# print("after alter obs space low is: ", env.observation_space.low.shape)
# print("after alter obs space high is: ", env.observation_space.high.shape)

actions = np.load('actions_doorkey.npy')

# print("shape of actions is: ", actions.shape)

for action in actions:
        print("shape of action is: ", action.shape)
        obs = env.reset()
        for ac in action:
                # print("shape of ac is: ", ac.shape)
                # print(ac)
                # print("ac is: ", ac)
                # print("shape of ac is: ", ac.shape)
                obs, rewards, done, info = env.step(ac)
                # print("obs is: ", obs)
                # print("action is: ", ac)
                env.render()
