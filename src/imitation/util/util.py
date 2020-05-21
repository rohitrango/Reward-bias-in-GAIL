import collections
import contextlib
import datetime
import functools
import os
from typing import Callable, Dict, Iterable, Optional, Tuple, Type, Union
import uuid

import gym
from gym.wrappers import TimeLimit
import numpy as np
import stable_baselines
from stable_baselines import bench
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.input import observation_input
from stable_baselines.common.policies import BasePolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
import tensorflow as tf

import gym_minigrid
from gym_minigrid import wrappers as mgwr

# TODO(adam): this should really be OrderedDict but that breaks Python
# See https://stackoverflow.com/questions/41207128/
LayersDict = Dict[str, tf.layers.Layer]


def make_unique_timestamp() -> str:
    """Timestamp, with random value added to avoid collisions (like
  human-readable uuid1)."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex
    return f"{timestamp}_{random_uuid}"


def make_vec_env(env_name: str,
                 n_envs: int = 8,
                 seed: int = 0,
                 parallel: bool = False,
                 log_dir: Optional[str] = None,
                 max_episode_steps: Optional[int] = None) -> VecEnv:
    """Returns a VecEnv initialized with `n_envs` Envs.

  Args:
      env_name: The Env's string id in Gym.
      n_envs: The number of duplicate environments.
      seed: The environment seed.
      parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
      log_dir: If specified, saves Monitor output to this directory.
      max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
          with this episode length. If not specified and `max_episode_steps`
          exists for this `env_name` in the Gym registry, uses the registry
          `max_episode_steps` for every TimeLimit wrapper (this automatic
          wrapper is the default behavior when calling `gym.make`). Otherwise
          the environments are passed into the VecEnv unwrapped.
  """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    spec = gym.spec(env_name)

    def make_env(i, this_seed):
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        env = spec.make()

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.seed(int(this_seed))

        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        elif spec.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=spec.max_episode_steps)

        # If minigrid, then use some postprocessing of the env
        keep_classes = ['goal', 'agent', 'wall', 'empty']
        drop_color = 1
        ename = env_name.lower()
        # for doorkey env
        if 'doorkey' in ename:
            keep_classes.extend(['door', 'key'])

        if 'gotodoor' in ename:
            keep_classes.extend(['door'])
            drop_color = 0

        # for redblue doors
        if 'redblue' in ename:
            keep_classes.extend(['door'])
            drop_color = 0

        if 'lava' in ename:
            keep_classes.extend(['lava'])
            drop_color = 1

        # would need to add an additional absorbing wrapper here.
        if env_name.startswith('MiniGrid'):
            env = mgwr.FullyObsWrapper(env)
            env = mgwr.ImgObsWrapper(env)
            env = mgwr.FullyObsOneHotWrapper(env, drop_color=drop_color, keep_classes=keep_classes, flatten=False)

            # env = AbsorbingWrapper(env)

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, 'monitor')
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f'{i:03d}')
        return MonitorPlus(env, log_path, allow_early_resets=True)

    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, (1 << 31) - 1, (n_envs,))
    env_fns = [functools.partial(make_env, i, s)
               for i, s in enumerate(env_seeds)]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method='forkserver')
    else:
        return DummyVecEnv(env_fns)


def init_rl(env: Union[gym.Env, VecEnv],
            model_class: Type[BaseRLModel] = stable_baselines.PPO2,
            policy_class: Type[BasePolicy] = MlpPolicy,
            **model_kwargs):
    """Instantiates a policy for the provided environment.

  Args:
      env: The (vector) environment.
      model_class: A Stable Baselines RL algorithm.
      policy_class: A Stable Baselines compatible policy network class.
      model_kwargs (dict): kwargs passed through to the algorithm.
        Note: anything specified in `policy_kwargs` is passed through by the
        algorithm to the policy network.

  Returns:
    An RL algorithm.
  """
    # 'init_rl_kwargs': {'ent_coef': 0.0,
    #               'learning_rate': 0.0003,
    #               'nminibatches': 32,
    #               'noptepochs': 10,
    #               'policy_class': 'CnnPolicy',
    #               'policy_kwargs': {
    #                                    'cnn_extractor': < function
    #               minigrid_extractor_small at
    #               0x7f76d270c290 >}, 'n_steps': 2048}

    print("initiating rl policy model kwargs: ", model_kwargs)
    return model_class(policy_class,
                       env, **model_kwargs)  # pytype: disable=not-instantiable


def build_mlp(hid_sizes: Iterable[int],
              name: Optional[str] = None,
              activation: Optional[Callable] = tf.nn.relu,
              initializer: Optional[Callable] = None,
              ) -> LayersDict:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    # Hidden layers
    for i, size in enumerate(hid_sizes):
        key = f"{name}_dense{i}"
        layer = tf.layers.Dense(size, activation=activation,
                                kernel_initializer=initializer,
                                name=key)  # type: tf.layers.Layer
        layers[key] = layer

    # Final layer
    layer = tf.layers.Dense(1, kernel_initializer=initializer,
                            name=f"{name}_dense_final")  # type: tf.layers.Layer
    layers[f"{name}_dense_final"] = layer

    return layers


def sequential(inputs: tf.Tensor,
               layers: LayersDict,
               ) -> tf.Tensor:
    """Applies a sequence of layers to an input."""
    output = inputs
    for layer in layers.values():
        output = layer(output)
    output = tf.squeeze(output, axis=1)
    return output


def build_inputs(observation_space: gym.Space,
                 action_space: gym.Space = None,
                 scale: bool = False,
                 ) -> Tuple[tf.Tensor, ...]:
    """Builds placeholders and processed input Tensors.

  Observation `obs_*` and `next_obs_*` placeholders and processed input
  tensors have shape `(None,) + obs_space.shape`.
  The action `act_*` placeholder and processed input tensors have shape
  `(None,) + act_space.shape`.

  Args:
    observation_space: The observation space.
    action_space: The action space.
    scale: Only relevant for environments with Box spaces. If True, then
      processed input Tensors are automatically scaled to the interval [0, 1].

  Returns:
    obs_ph: Placeholder for old observations.
    act_ph: Placeholder for actions.
    next_obs_ph: Placeholder for new observations.
    obs_inp: Network-ready float32 Tensor with processed old observations.
    act_inp: Network-ready float32 Tensor with processed actions.
    next_obs_inp: Network-ready float32 Tensor with processed new observations.
  """
    # if not use_terminal_state:
    obs_ph, obs_inp = observation_input(observation_space,
                                        name="obs", scale=scale)
    act_ph, act_inp = observation_input(action_space, name="act", scale=scale)
    next_obs_ph, next_obs_inp = observation_input(observation_space,
                                                  name="next_obs", scale=scale)
    return obs_ph, act_ph, next_obs_ph, obs_inp, act_inp, next_obs_inp

    # else:
    #     obs_ph, obs_inp = observation_input(observation_space,
    #                                         name="obs", scale=scale)
    #     return obs_ph, obs_inp


@contextlib.contextmanager
def make_session(close_on_exit: bool = True, **kwargs):
    """Context manager for a TensorFlow session.

  The session is associated with a newly created graph. Both session and
  graph are set as default. The session will be closed when exiting this
  context manager.

  Args:
    close_on_exit: If True, closes the session upon leaving the context manager.
    kwargs: passed through to `tf.Session`.

  Yields:
    (graph, session) where graph is a `tf.Graph` and `session` a `tf.Session`.
  """
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session(graph=graph, **kwargs)
        try:
            with session.as_default():
                yield graph, session
        finally:
            if close_on_exit:
                session.close()


def docstring_parameter(*args, **kwargs):
    """Treats the docstring as a format string, substituting in the arguments."""

    def helper(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return helper


class MonitorPlus(bench.Monitor):
    """Augments Monitor by recording raw rewards and observations.

  Adds two new keys to `info["episode"]` (which is returned whenever done=True),
  "obs" and "rews", which hold the raw observations and rewards seen during
  this episode.
  """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs = super().reset()
        self._obs = [new_obs]
        self._rews = []
        return new_obs

    def step(self, action):
        obs, rew, done, info = super().step(action)
        self._obs.append(obs)
        self._rews.append(rew)

        if done:
            ep_info = info.get('episode')
            assert isinstance(ep_info, dict)
            ep_info["obs"] = np.stack(self._obs)
            ep_info["rews"] = np.stack(self._rews)

        return obs, rew, done, info
