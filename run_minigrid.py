import stable_baselines
import tensorflow as tf
import argparse

import gym_minigrid
import gym
import os

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines.deepq.policies import CnnPolicy as CnnDQNPolicy
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.bench import Monitor
from stable_baselines import DQN, PPO2, HER, PPO2
# Custom CNN-LSTM Policy for MiniGrid
from stable_baselines.common.policies import minigrid_extractor, minigrid_extractor_small
from stable_baselines.gail import generate_expert_traj

from gym_minigrid import wrappers as mgwr

parser = argparse.ArgumentParser()
# Env params
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--fullobs', type=int, default=1)
parser.add_argument('--algo', type=str, required=True)
parser.add_argument('--total_steps', type=int, default=int(1e5))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--see_through_walls', type=int, default=0)
parser.add_argument('--policy', type=str, default='cnn')
parser.add_argument('--n_lstm', type=int, default=256)
parser.add_argument('--train', type=int, default=1)

# Load and save models
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--load_model_name', type=str, default=None)

# Save trajectories
parser.add_argument('--n_episodes', type=int, default=100)
parser.add_argument('--save_expert', type=int, default=0)

# Architecture params
parser.add_argument('--layer_norm', type=int, default=0,)

# Visualization and evaluation params
parser.add_argument('--viz', type=int, default=0)

args = parser.parse_args()
args.see_through_walls = bool(args.see_through_walls)

# Check for consistency
assert args.train + args.viz + args.save_expert == 1

# Get a suffix
if not args.fullobs:
    suffix = '_lstm'
else:
    suffix = ''

# Get current date and time
import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

# Check for args model name
if args.model_name is None:
    args.model_name = '{}_{}_{}_{}'.format(args.env, args.algo, suffix, date)

# Modify the classes to keep
keep_classes = ['agent', 'goal', 'wall', 'empty']
if 'door' in args.env.lower():
    keep_classes.extend(['door', 'key'])
if 'unlock' in args.env.lower():
    keep_classes.extend(['door', 'key'])

# Append classes to keep
if keep_classes is not None:
    if not args.fullobs and not args.see_through_walls:
        keep_classes.append('unseen')
log_dir = '/serverdata/rohit/reward_bias/logs/{}/{}/'.format(args.model_name, args.seed)
print(args)
print("Saving to {} ...\n".format(log_dir))


def load_env(i=0):

    def _load_env():
        # Load the required env
        envname = args.env
        env = gym.make(envname)
        env.see_through_walls = args.see_through_walls
        # Make a directory if doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        # Append if exists already
        append = args.load_model_name == args.model_name
        # Create monitor for env
        if args.train:
            env = Monitor(env, log_dir + '{}'.format(i), allow_early_resets=True, append=append)
        # MiniGrid
        if args.env.startswith('MiniGrid') and args.fullobs:
            env = mgwr.FullyObsWrapper(env)
            env = mgwr.ImgObsWrapper(env)
            env = mgwr.FullyObsOneHotWrapper(env, drop_color=1, keep_classes=keep_classes, flatten=False)
            # Make a goal policy for HER
            if args.algo == 'her':
                env = mgwr.GoalPolicyWrapper(env)
        elif args.env.startswith('MiniGrid') and not args.fullobs:
            env = mgwr.ImgObsWrapper(env)
            env = mgwr.FullyObsOneHotWrapper(env, drop_color=1, keep_classes=keep_classes, flatten=False)
            # else if its a minigrid but not a fullyobs
        return env

    return _load_env


def get_policy():
    algo = args.algo
    if algo == 'dqn':
        policy = CnnDQNPolicy
    elif algo == 'her':
        policy = CnnDQNPolicy
    elif algo == 'ppo':
        if args.policy == 'cnn':
            policy = CnnPolicy if args.fullobs else CnnLstmPolicy
        elif args.policy == 'att':
            raise NotImplementedError
    else:
        raise NotImplementedError
    return policy


def main():
    # Main function will define a policy, and run the algorithm
    num_envs = 16
    if not args.train:
        num_envs = 1

    # Check for algorithms and envs
    if args.algo != 'her':
        enva = DummyVecEnv([load_env(i) for i in range(num_envs)])
        #env = VecNormalize(enva, training=args.train)
        env = enva
    else:
        env = load_env()()

    # Check for envs loading
    if args.load_model_name:
        #env.load_running_average('/serverdata/rohit/reward_bias/{}_{}_env'.format(args.load_model_name, args.seed))
        pass

    print(env.observation_space)
    print(env.action_space)

    # Hack to use the smaller conv net if the visibility is low
    if min(env.observation_space.shape[:2]) < 9:
        extractor = minigrid_extractor_small
    else:
        extractor = minigrid_extractor
    # Get Policy from given parameters
    policy = get_policy()
    policy_kwargs = {
            'cnn_extractor': extractor,
        }

    # Add lstm
    if not args.fullobs:
        policy_kwargs['n_lstm'] = args.n_lstm
    print(policy_kwargs)

    # Check for algos
    algo = args.algo
    if algo == 'ppo':
        # Load custom trained model if needed
        if args.load_model_name:
            model = PPO2.load('/serverdata/rohit/reward_bias/{}_{}'.format(args.load_model_name, args.seed), env=env)
        else:
            model = PPO2(policy, env, verbose=1, seed=args.seed, \
                policy_kwargs=policy_kwargs, tensorboard_log=log_dir)
    else:
        raise NotImplementedError

    ##############################################################
    ## Actual learning or viz
    ##############################################################
    # learn
    if args.train:
        model.learn(total_timesteps=args.total_steps)
        model.save('/serverdata/rohit/reward_bias/{}_{}'.format(args.model_name, args.seed))
        # also save vec env params
        os.makedirs('/serverdata/rohit/reward_bias/{}_{}_env'.format(args.model_name, args.seed), exist_ok=True)
        try:
            env.save_running_average('/serverdata/rohit/reward_bias/{}_{}_env'.format(args.model_name, args.seed))
        except:
            pass
    elif args.viz:
        # Visualize the env and the agent in it
        for _ in range(20):
            obs = env.reset()
            while True:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                print(rewards[0], dones[0])
                env.envs[0].render('human')
                if dones[0]:
                    break

    elif args.save_expert:
        # Save expert trajectories
        save_path = os.path.join(model, '/serverdata/rohit/reward_bias/experts/{}_{}'.format(args.load_model_name, args.seed))
        image_folder = os.path.join(model, '/serverdata/rohit/reward_bias/experts/{}_{}_images'.format(args.load_model_name, args.seed))
        generate_expert_traj(model, save_path, n_timesteps=0, n_episodes=args.n_episodes, image_folder=image_folder)


if __name__ == "__main__":
    main()
