from stable_baselines.common.policies import MlpPolicy, CnnPolicy, minigrid_extractor_small, FeedForwardPolicy
from stable_baselines import PPO1
import gym
from minigrid_gail_wrapper import MiniGridGailWrapper
import tensorflow as tf 

def callback(a,b):
	# print("callback called")
	print((a.keys()))
	print(b.keys())
	model.save("ppo1_cartpole")


# env = gym.make('CartPole-v1')

import gym_minigrid, gym
from gym_minigrid import wrappers as mgwr
import numpy as np

keep_classes = ['goal', 'agent', 'wall', 'empty']
drop_color =1
env = gym.make('MiniGrid-Empty-6x6-v0')
print("initially: ", env.actions)
# print("env obs space: ", env.observation_space)
# print(env.observation_space)
keep_classes.extend(['door', 'key'])

# print("state is: ", env.observation)
env = mgwr.FullyObsWrapper(env)
env = mgwr.ImgObsWrapper(env)
env = mgwr.FullyObsOneHotWrapper(env, drop_color=drop_color, keep_classes=keep_classes, flatten=False)
# env = MiniGridGailWrapper(env)

# print("steps: ", env.max_steps)

# model = PPO1(FeedForwardPolicy(tf.compat.v1.get_default_session(), env.observation_space, env.action_space, 1, 1,
#  None, False, cnn_extractor=minigrid_extractor_small), env, verbose=1)
model = PPO1(CnnPolicy, env, verbose=1)

print("learn called")
model.learn(total_timesteps=25000)
model.save("ppo1_minigrid")

# for i in range(25000):
model = PPO1.load("ppo1_minigrid")
# model.set_env(env)
# model.learn(total_timesteps=1, callback=callback)

model = PPO1.load("ppo1_minigrid")
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

