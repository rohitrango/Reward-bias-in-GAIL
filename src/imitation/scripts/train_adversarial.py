"""Train GAIL or AIRL and plot its output.

Can be used as a CLI script, or the `train_and_plot` function can be called
directly.
"""

import os
import os.path as osp
import pickle
from typing import Optional
import time
from sacred.observers import FileStorageObserver
import tensorflow as tf
from stable_baselines.common.vec_env import VecEnv, VecNormalize

from imitation.algorithms.adversarial import init_trainer
import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_ex
import imitation.util as util
import imitation.util.sacred as sacred_util


def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    try:
        trainer.discrim.save(os.path.join(save_path, "discrim"))
        # TODO(gleave): unify this with the saving logic in data_collect?
        # (Needs #43 to be merged before attempting.)
        serialize.save_stable_model(os.path.join(save_path, "gen_policy"),
                                    trainer.gen_policy,
                                    trainer.venv_train_norm)
    except:
        pass


@train_ex.main
def train(_run,
          _seed: int,
          env_name: str,
          rollout_path: str,
          normalize: bool,
          normalize_kwargs: dict,
          n_expert_demos: Optional[int],
          log_dir: str,
          init_trainer_kwargs: dict,
          total_timesteps: int,
          n_episodes_eval: int,
          init_tensorboard: bool,
          checkpoint_interval: int) -> dict:
    """Train an adversarial-network-based imitation learning algorithm.

  Plots (turn on using `plot_interval > 0`):
    - Plot discriminator loss during discriminator training steps in blue and
      discriminator loss during generator training steps in red.
    - Plot the performance of the generator policy versus the performance of
      a random policy. Also plot the performance of an expert policy if that is
      provided in the arguments.

  Checkpoints:
    - DiscrimNets are saved to f"{log_dir}/checkpoints/{step}/discrim/",
      where step is either the training epoch or "final".
    - Generator policies are saved to
      f"{log_dir}/checkpoints/{step}/gen_policy/".

  Args:
    _seed: Random seed.
    env_name: The environment to train in.
    rollout_path: Path to pickle containing list of Trajectories. Used as
      expert demonstrations.
    n_expert_demos: The number of expert trajectories to actually use
      after loading them from `rollout_path`.
      If None, then use all available trajectories.
      If `n_expert_demos` is an `int`, then use exactly `n_expert_demos`
      trajectories, erroring if there aren't enough trajectories. If there are
      surplus trajectories, then use the
      first `n_expert_demos` trajectories and drop the rest.
    log_dir: Directory to save models and other logging to.

    init_trainer_kwargs: Keyword arguments passed to `init_trainer`,
      used to initializtrain_adversariale the trainer.
    total_timesteps: The number of transitions to sample from the environment
      during training.
    n_episodes_eval: The number of episodes to average over when calculating
      the average episode reward of the imitation policy for return.

    plot_interval: The number of epochs between each plot. If negative,
      then plots are disabled. If zero, then only plot at the end of training.
    n_plot_episodes: The number of episodes averaged over when
      calculating the average episode reward of a policy for the performance
      plots.
    extra_episode_data_interval: Usually mean episode rewards are calculated
      immediately before every plot. Set this parameter to a nonnegative number
      to also add episode reward data points every
      `extra_episodes_data_interval` epochs.
    show_plots: Figures are always saved to `f"{log_dir}/plots/*.png"`. If
      `show_plots` is True, then also show plots as they are created.
    init_tensorboard: If True, then write tensorboard logs to `{log_dir}/sb_tb`.

    checkpoint_interval: Save the discriminator and generator models every
      `checkpoint_interval` epochs and after training is complete. If 0,
      then only save weights after training is complete. If <0, then don't
      save weights at all.

  Returns:
    A dictionary with two keys. "imit_stats" gives the return value of
      `rollout_stats()` on rollouts test-reward-wrapped
      environment, using the final policy (remember that the ground-truth reward
      can be recovered from the "monitor_return" key). "expert_stats" gives the
      return value of `rollout_stats()` on the expert demonstrations loaded from
      `rollout_path`.
  """

    # print("use terminal is: ", use_terminal_state_disc)
    print("function called..")
    print("logdir: ", log_dir)
    print("rollouts: ", rollout_path)

    print("kwargs: ", init_trainer_kwargs)
    print("kwargs: ", init_trainer_kwargs.keys())

    total_timesteps = int(total_timesteps)
    print("_run is: ", _run)
    tf.logging.info("Logging to %s", log_dir)

    time.sleep(10.0)
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    # Calculate stats for expert rollouts. Used for plot and return value.
    # load expert rollouts.

    # the rollouts might need to be wrapped.
    with open(rollout_path, "rb") as f:
        expert_trajs = pickle.load(f)

    # only use a certain subset of the trajectories
    if n_expert_demos is not None:
        assert len(expert_trajs) >= n_expert_demos
        expert_trajs = expert_trajs[:n_expert_demos]

    # get stats about length, returns etc from the expert data.
    # this is similar to the do_rollout function.
    expert_stats = util.rollout.rollout_stats(expert_trajs)
    #
    # kwargs: {'discrim_kwargs': {'build_discrim_net_kwargs': {'cnn_extractor': < function minigrid_extractor_small at
    #                             0x7f76d270c290 >}, 'reward_type': 'positive'},
    # 'init_rl_kwargs': {'ent_coef': 0.0,
    #               'learning_rate': 0.0003,
    #               'nminibatches': 32,
    #               'noptepochs': 10,
    #               'policy_class': 'CnnPolicy',
    #               'policy_kwargs': {
    #                                    'cnn_extractor': < function
    #               minigrid_extractor_small at
    #               0x7f76d270c290 >}, 'n_steps': 2048}, 'use_gail': True, 'trainer_kwargs': {
    #     'disc_minibatch_size': 512, 'gen_replay_buffer_capacity': 16384,
    #     'disc_batch_size': 16384}, 'num_vec': 8, 'parallel': True, 'max_episode_steps': None, 'scale': True, 'reward_kwargs': {
    #     'theta_units': [32, 32], 'phi_units': [32, 32]}, 'use_bc': False}

    # gives a graph and a session
    with util.make_session():
        if init_tensorboard:
            sb_tensorboard_dir = osp.join(log_dir, "sb_tb")
            kwargs = init_trainer_kwargs
            kwargs["init_rl_kwargs"] = kwargs.get("init_rl_kwargs", {})
            kwargs["init_rl_kwargs"]["tensorboard_log"] = sb_tensorboard_dir

        trainer = init_trainer(env_name, expert_trajs,
                               seed=_seed, log_dir=log_dir,
                               normalize=normalize, normalize_kwargs=normalize_kwargs,
                               **init_trainer_kwargs)

        def callback(epoch):
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

        if init_trainer_kwargs['use_bc']:
            trainer.train(n_epochs=total_timesteps, on_epoch_end=callback)
        else:
            trainer.train(total_timesteps, callback)

        # Save final artifacts.
        if checkpoint_interval >= 0:
            save(trainer, os.path.join(log_dir, "checkpoints", "final"))

        # Final evaluation of imitation policy. 
        results = {}
        sample_until_eval = util.rollout.min_episodes(n_episodes_eval)
        trajs = util.rollout.generate_trajectories(trainer.gen_policy,
                                                   trainer.venv_test,
                                                   sample_until=sample_until_eval)
        results["imit_stats"] = util.rollout.rollout_stats(trajs)
        results["expert_stats"] = expert_stats
        return results


def main_console():
    print(osp.join('output', 'sacred', 'train'))
    time.sleep(10.0)
    observer = FileStorageObserver.create(osp.join('output', 'sacred', 'train'))
    train_ex.observers.append(observer)
    train_ex.run_commandline()


if __name__ == "__main__":
    print("script called")
    main_console()
