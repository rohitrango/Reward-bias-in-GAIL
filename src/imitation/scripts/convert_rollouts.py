import pickle
import numpy as np
import gym
import imitation.envs.examples
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
import imitation.util as util
from imitation.util.reward_wrapper import RewardVecEnvWrapper
import imitation.util.sacred as sacred_util
import h5py

"""
A script to convert the rollouts generated from Reward bias repo to a format which is compatible with DAC

"""

# load the desired trajectory by changing the path here.
trajectories = pickle.load(open("src/imitation/scripts/Hopper-v2/HopperPPO/rollouts/final.pkl", "rb"))

# name of the file saved file after conversion
out_file = 'Hopper_trpo.h5'

observations, actions, length, rewards_traj = [], [], [], []


# this can be changed the DAC paper's results are on four expert trajectories for each environment.
num_trajectories = 4

# ep_length 
ep_length = 1000

i = 0
while len(actions) < num_trajectories:
    actions_episode = trajectories[i].acts
    obs_episode = trajectories[i].obs 
    rewards_episode = trajectories[i].rews
    i += 1
    if len(actions_episode) == ep_length:
	    length_episode = len(rewards_episode)
	    print("len is: ", length_episode)
	    length.append(length_episode)

	    observations.append(obs_episode)
	    actions.append(actions_episode)
	    rewards_traj.append(rewards_episode)

observations = np.array(observations)
actions = np.array(actions)
length = np.array(length)
rewards_traj = np.array(rewards_traj)

print("shape of observations: ", observations.shape)
print("shape of actions: ", actions.shape)
print("shape of length: ", length.shape)
print("shape of rewards_traj: ", rewards_traj.shape)

hf = h5py.File(out_file, 'w')
hf.create_dataset('a_B_T_Da', data=actions)
hf.create_dataset('len_B', data=length)
hf.create_dataset('r_B_T', data=rewards_traj)

hf.create_dataset('obs_B_T_Do', data=observations)
hf.close()