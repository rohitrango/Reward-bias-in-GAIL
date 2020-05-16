import contextlib
import os
import os.path as osp
from typing import Optional

from sacred.observers import FileStorageObserver
from stable_baselines.common.vec_env import VecNormalize
import stable_baselines.logger as sb_logger
import tensorflow as tf

import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
import imitation.util as util
from imitation.util.reward_wrapper import RewardVecEnvWrapper
import imitation.util.sacred as sacred_util


@expert_demos_ex.main
def rollouts_and_policy(
  _run,
  _seed: int,
  env_name: str,
  total_timesteps: int,
  *,
  log_dir: str,
  num_vec: int,
  parallel: bool,
  max_episode_steps: Optional[int],
  normalize: bool,
  normalize_kwargs: dict,
  init_rl_kwargs: dict,

  n_episodes_eval: int,

  reward_type: Optional[str],
  reward_path: Optional[str],

  rollout_save_interval: int,
  rollout_save_final: bool,
  rollout_save_n_timesteps: Optional[int],
  rollout_save_n_episodes: Optional[int],

  policy_save_interval: int,
  policy_save_final: bool,

  init_tensorboard: bool,
) -> dict:


  """

  Trains an expert policy from scratch and saves the rollouts and policy.

  Checkpoints:
    At applicable training steps `step` (where step is either an integer or
    "final"):

      - Policies are saved to `{log_dir}/policies/{step}/`.
      - Rollouts are saved to `{log_dir}/rollouts/{step}.pkl`.

  Args:
      env_name: The gym.Env name. Loaded as VecEnv.
      total_timesteps: Number of training timesteps in `model.learn()`.
      log_dir: The root directory to save metrics and checkpoints to.
      num_vec: Number of environments in VecEnv.
      parallel: If True, then use DummyVecEnv. Otherwise use SubprocVecEnv.
      max_episode_steps: If not None, then environments are wrapped by
          TimeLimit so that they have at most `max_episode_steps` steps per
          episode.
      normalize: If True, then rescale observations and reward.
      normalize_kwargs: kwargs for `VecNormalize`.
      init_rl_kwargs: kwargs for `init_rl`.

      n_episodes_eval: The number of episodes to average over when calculating
          the average ground truth reward return of the final policy.

      reward_type: If provided, then load the serialized reward of this type,
          wrapping the environment in this reward. This is useful to test
          whether a reward model transfers. For more information, see
          `imitation.rewards.serialize.load_reward`.
      reward_path: A specifier, such as a path to a file on disk, used by
          reward_type to load the reward model. For more information, see
          `imitation.rewards.serialize.load_reward`.

      rollout_save_interval: The number of training updates in between
          intermediate rollout saves. If the argument is nonpositive, then
          don't save intermediate updates.
      rollout_save_final: If True, then save rollouts right after training is
          finished.
      rollout_save_n_timesteps: The minimum number of timesteps saved in every
          file. Could be more than `rollout_save_n_timesteps` because
          trajectories are saved by episode rather than by transition.
          Must set exactly one of `rollout_save_n_timesteps`
          and `rollout_save_n_episodes`.
      rollout_save_n_episodes: The number of episodes saved in every
          file. Must set exactly one of `rollout_save_n_timesteps` and
          `rollout_save_n_episodes`.

      policy_save_interval: The number of training updates between saves. Has
          the same semantics are `rollout_save_interval`.
      policy_save_final: If True, then save the policy right after training is
          finished.

      init_tensorboard: If True, then write tensorboard logs to {log_dir}/sb_tb
          and "output/summary/...".

  Returns:
    The return value of `rollout_stats()` using the final policy.
  """
  os.makedirs(log_dir, exist_ok=True)
  sacred_util.build_sacred_symlink(log_dir, _run)

  sample_until = util.rollout.make_sample_until(rollout_save_n_timesteps,
                                                rollout_save_n_episodes)
  eval_sample_until = util.rollout.min_episodes(n_episodes_eval)

  with util.make_session():
    tf.logging.set_verbosity(tf.logging.INFO)
    util.logger.configure(folder=osp.join(log_dir, 'rl'),
                          format_strs=['tensorboard', 'stdout'])

    rollout_dir = osp.join(log_dir, "rollouts")
    policy_dir = osp.join(log_dir, "policies")
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    if init_tensorboard:
      sb_tensorboard_dir = osp.join(log_dir, "sb_tb")
      init_rl_kwargs["tensorboard_log"] = sb_tensorboard_dir

    venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                             parallel=parallel, log_dir=log_dir,
                             max_episode_steps=max_episode_steps)

    log_callbacks = []
    with contextlib.ExitStack() as stack:
      if reward_type is not None:
        reward_fn_ctx = load_reward(reward_type, reward_path, venv)
        reward_fn = stack.enter_context(reward_fn_ctx)
        venv = RewardVecEnvWrapper(venv, reward_fn)
        log_callbacks.append(venv.log_callback)
        tf.logging.info(
            f"Wrapped env in reward {reward_type} from {reward_path}.")

      vec_normalize = None
      if normalize:
        venv = vec_normalize = VecNormalize(venv, **normalize_kwargs)

      policy = util.init_rl(venv, verbose=1, **init_rl_kwargs)

      # Make callback to save intermediate artifacts during training.
      step = 0

      def callback(locals_: dict, _) -> bool:
        nonlocal step
        step += 1
        policy = locals_['self']

        # TODO(adam): make logging frequency configurable
        for callback in log_callbacks:
          callback(sb_logger)

        if rollout_save_interval > 0 and step % rollout_save_interval == 0:
          save_path = osp.join(rollout_dir, f"{step}.pkl")
          util.rollout.save(save_path, policy, venv, sample_until)
        if policy_save_interval > 0 and step % policy_save_interval == 0:
          output_dir = os.path.join(policy_dir, f'{step:05d}')
          serialize.save_stable_model(output_dir, policy, vec_normalize)

      policy.learn(total_timesteps, callback=callback)

      # Save final artifacts after training is complete.
      if rollout_save_final:
        save_path = osp.join(rollout_dir, "final.pkl")
        util.rollout.save(save_path, policy, venv, sample_until)
      if policy_save_final:
        output_dir = os.path.join(policy_dir, "final")
        serialize.save_stable_model(output_dir, policy, vec_normalize)

      # Final evaluation of expert policy.
      trajs = util.rollout.generate_trajectories(
          policy, venv, eval_sample_until)
      stats = util.rollout.rollout_stats(trajs)

  return stats


@expert_demos_ex.command
@util.make_session()
def rollouts_from_policy(
  _run,
  _seed: int,
  *,
  num_vec: int,
  rollout_save_n_timesteps: int,
  rollout_save_n_episodes: int,
  log_dir: str,
  policy_path: str,
  policy_type: str,
  env_name: str,
  parallel: bool,
  rollout_save_path: str,
  max_episode_steps: Optional[int],
) -> None:
  """Loads a saved policy and generates rollouts.

  Unlisted arguments are the same as in `rollouts_and_policy()`.

  Args:
      policy_type: Argument to `imitation.policies.serialize.load_policy`.
      policy_path: Argument to `imitation.policies.serialize.load_policy`.
      rollout_save_path: Rollout pickle is saved to this path.
  """
  os.makedirs(log_dir, exist_ok=True)
  sacred_util.build_sacred_symlink(log_dir, _run)

  sample_until = util.rollout.make_sample_until(rollout_save_n_timesteps,
                                                rollout_save_n_episodes)

  venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                           parallel=parallel, log_dir=log_dir,
                           max_episode_steps=max_episode_steps)

  with serialize.load_policy(policy_type, policy_path, venv) as policy:
    print(policy)
    util.rollout.save(rollout_save_path, policy, venv, sample_until)


def main_console():
  observer = FileStorageObserver.create(
      osp.join('/serverdata/sid/reward_bias/imitation', 'output', 'sacred', 'expert_demos'))
  expert_demos_ex.observers.append(observer)
  expert_demos_ex.run_commandline()


if __name__ == "__main__":
  main_console()
