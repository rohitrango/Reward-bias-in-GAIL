"""Common wrapper for adding custom reward values to an environment."""
import collections
from typing import Callable

import numpy as np
from stable_baselines.common import vec_env

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    np.ndarray]


class RewardVecEnvWrapper(vec_env.VecEnvWrapper):
    def __init__(self,
                 venv: vec_env.VecEnv,
                 reward_fn: RewardFn,
                 ep_history: int = 100,
                 terminal_reward: RewardFn = None):
        """A RewardVecEnvWrapper uses a provided reward_fn to replace
    the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization. A tricky part
    about this class is keeping track of the most recent observation from each
    environment.

    Will also include the previous reward given by the inner VecEnv in the
    returned info dict under the `wrapped_env_rew` key.

    Args:
        venv: The VecEnv to wrap.
        reward_fn: A function that wraps takes in vectorized transitions
            (obs, act, next_obs) a vector of episode timesteps, and returns a
            vector of rewards.
        ep_history: The number of episode rewards to retain for computing
            mean reward.
    """
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self.episode_rewards = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn
        self.terminal_reward = terminal_reward
        print("terminal reward is: ", terminal_reward)
        self.reset()

    def log_callback(self, logger):
        """Logs mean reward over the last `ep_history` episodes."""
        if len(self.episode_rewards) == 0:
            return
        mean = sum(self.episode_rewards) / len(self.episode_rewards)
        logger.logkv("eprewmean_wrapped", mean)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        self._step_counter = np.zeros((self.num_envs,), dtype='int')
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # print("dones is: ", dones)
        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                # print("when done single_obs is: ", single_obs)
                single_obs = single_infos['terminal_observation']
                # print("after using terminal obs info: ", single_obs)

            obs_fixed.append(single_obs)
        obs_fixed = np.stack(obs_fixed)

        rews = self.reward_fn(self._old_obs,
                              self._actions,
                              obs_fixed,
                              self._step_counter)

        # if np.all(dones == True) and self.terminal_reward is not None:
        rews_terminal = self.terminal_reward(self._old_obs,
                                             self._actions,
                                             obs_fixed,
                                             self._step_counter)

        print("shape of rews_terminal is: ", rews_terminal.shape)
        print("adding terminal reward")
        rews += np.asarray(dones, dtype='bool').reshape((len(dones),)) * rews_terminal

        #

        # print("shape of rews is: ", rews.shape)
        # print("shape of obs is: ", obs.shape)
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype='bool').reshape((len(dones),))

        # print("done mask is: ", done_mask)
        self._step_counter += 1
        self._step_counter[done_mask] = 0

        # Update statistics
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict['wrapped_env_rew'] = old_rew
        return obs, rews, dones, infos
