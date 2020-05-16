import pickle
import numpy as np
import gym
import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
import imitation.util as util
from imitation.util.reward_wrapper import RewardVecEnvWrapper
import imitation.util.sacred as sacred_util
import h5py

trajectories = pickle.load(open("src/imitation/scripts/Hopper-v2/HopperPPO/rollouts/final.pkl", "rb"))
# trajectories = pickle.load(open("src/imitation/scripts/Hopper-v2/HopperPPO/policies/final/model.pkl", "rb"))

# print(type(trajectories))
# print(len(trajectories))
# print(type(trajectories[1]))
# print(trajectories[1].acts)

# env = gym.make('Hopper-v2')

# actions = trajectories[i].acts
# obs = trajectories[i].obs 
# rews = trajectories[i].rews

# obs = env.reset()
# for action in actions:
#     # action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()

observations, actions, length, rewards_traj = [], [], [], []


num_trajectories = 4

i = 0
while len(actions) < num_trajectories:
    print("traj: ", i)
    # obs = env.reset()
    # obs_episode = []
    # actions_episode = []

    # length_episode = 0
    # rewards_episode = []

    actions_episode = trajectories[i].acts
    print("type of actions_episode is: ", actions_episode.shape)
    obs_episode = trajectories[i].obs 
    print("type of observations is: ", obs_episode.shape)
    rewards_episode = trajectories[i].rews
    print("type of rewards is: ", rewards_episode.shape	)
    i += 1
    if len(actions_episode) == 1000:
    # done = False

    # while not done:
    #     obs_episode.append(obs)
    #     action, _states = model.predict(obs)
    #     actions_episode.append(action)

    #     obs, rewards, done, info = env.step(action)
    #     # print(rewards)
    #     rewards_episode.append(rewards)
    #     # env.render()

        # print("done is: ", dones)

	    length_episode = len(rewards_episode)
	    print("len is: ", length_episode)
	    length.append(length_episode)

	    observations.append(obs_episode)
	    actions.append(actions_episode)
	    # print(rewards_episode)
	    rewards_traj.append(rewards_episode)
    # env.render()


observations = np.array(observations)
actions = np.array(actions)
length = np.array(length)
rewards_traj = np.array(rewards_traj)

print("shape of observations: ", observations.shape)
print("shape of actions: ", actions.shape)
print("shape of length: ", length.shape)
print("shape of rewards_traj: ", rewards_traj.shape)

hf = h5py.File('Hopper_trpo.h5', 'w')

print("file declaration done")

hf.create_dataset('a_B_T_Da', data=actions)

print("action done")
hf.create_dataset('len_B', data=length)
print("length dones")

hf.create_dataset('r_B_T', data=rewards_traj)

hf.create_dataset('obs_B_T_Do', data=observations)

print("obs done")
print(hf.keys())

hf.close()