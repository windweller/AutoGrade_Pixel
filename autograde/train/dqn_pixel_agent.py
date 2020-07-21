"""
Use a simpler wrapper to train DQN style
"""

import os
import gym
import argparse
import tensorflow as tf
from stable_baselines import DQN
from stable_baselines.common.atari_wrappers import ScaledFloatFrame, WarpFrame, MaxAndSkipEnv, FrameStack

from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, \
    StopTrainingOnRewardThreshold
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines.common.evaluation import evaluate_policy

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame

from collections import deque

import numpy as np


def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale

    (400, 400, 3)
    ValueError: cannot reshape array of size 480000 into shape (210,160,3)

    100800 (200x200x3)

    """
    # state = np.reshape(state, [210, 160, 3]).astype(np.float32)
    state = np.reshape(state, [84, 84, 3]).astype(np.float32)

    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # karpathy
    # state = state[35:195]  # crop
    # state = state[::2, ::2]  # downsample by factor of 2
    # state = state[::4, ::4]  # downsample by factor of 4

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)


class GrayScaleFrame(gym.ObservationWrapper):
    def __init__(self, env):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.

        Does not cast GRAYSCALE (because it leads to weird image)
        This wrapper is ONLY used with recurrent policy

        :param env: (Gym Environment) the environment
        """
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1),
                                                dtype=env.observation_space.dtype)

    def observation(self, frame):
        """
        returns the current observation from a frame

        :param frame: ([int] or [float]) environment frame
        :return: ([int] or [float]) the observation
        """
        frame = greyscale(frame)
        return frame


def get_env_fn(program, reward_type, reward_shaping, max_skip, num_ball_to_win):
    # we add all necessary wrapper here
    def make_env():
        env = BouncePixelEnv(program, reward_type, reward_shaping, num_ball_to_win=num_ball_to_win, finish_reward=0)
        # env = WarpFrame(env)
        # env = ScaledFloatFrame(env)
        env = ResizeFrame(env)
        env = GrayScaleFrame(env)
        env = FrameStack(env, 4)
        env = MaxAndSkipEnv(env, skip=max_skip)
        # env = WarpFrame(env)
        # env = ScaledFloatFrame(env)
        return env

    return make_env


def make_general_env(program, max_skip, num_envs, reward_type, reward_shaping, num_ball_to_win):
    base_env_fn = get_env_fn(program, reward_type, reward_shaping, max_skip, num_ball_to_win)

    env = base_env_fn()

    return env


def test_observations():
    # ========= Execute some random actions and observe ======
    program = Program()
    program.set_correct()
    env = make_general_env(program, 1, 1, ONLY_SELF_SCORE, False, num_ball_to_win=1)

    import numpy as np
    from autograde.rl_envs.utils import SmartImageViewer

    viewer = SmartImageViewer()

    obs = env.reset()
    for i in range(1000):
        action = np.random.randint(env.action_space.n, size=1)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        obs = obs.squeeze(0)

        viewer.imshow(obs)
        if rewards[0] != 0:
            print(rewards)
    env.close()


def main():
    program = Program()
    program.set_correct_with_theme()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1)
    print("Number of environments: {}".format(env.num_envs))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=100000,
                                                 save_path="./saved_models/dqn_self_minus_finish_reward/",
                                                 name_prefix="DQN_Nature_CNN_default")

        model = DQN(CnnPolicy, env, learning_rate=5e-4, gamma=0.99, buffer_size=50000, target_network_update_freq=10000,
                    learning_starts=50000, train_freq=4,
                    prioritized_replay=True,
                    verbose=1, tensorboard_log="./tensorboard_self_minus_finish_log/")

        print("Evaluating initial performance...")
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1)
        mean_reward, std_reward = evaluate_policy(model, single_env, n_eval_episodes=10)
        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        print("Training starts!")
        model.learn(total_timesteps=2000000, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='DQN-Nature-CNN')  # 10M

        model.save("./saved_models/dqn_self_minus_finish_reward")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1)
        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_policy(model, single_env, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()


if __name__ == '__main__':
    main()
