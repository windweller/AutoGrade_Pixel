import os
import argparse
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.atari_wrappers import ScaledFloatFrame, WarpFrame
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.policies import LstmPolicy

import numpy as np
from gym.wrappers import TimeLimit

from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, \
    StopTrainingOnRewardThreshold

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame

def get_env_fn(program, reward_type, reward_shaping, num_ball_to_win, max_steps=1000, finish_reward=0):
    # we add all necessary wrapper here
    def make_env():
        env = BouncePixelEnv(program, reward_type, reward_shaping, num_ball_to_win=num_ball_to_win,
                             finish_reward=finish_reward)
        # env = WarpFrame(env)
        # env = ScaledFloatFrame(env)
        env = ResizeFrame(env)
        # env = ScaledFloatFrame(env)
        env = TimeLimit(env, max_episode_steps=max_steps)  # 20 seconds  # no skipping, it should be 3000. With skip, do 3000 / skip.
        return env

    return make_env

def make_general_env(program, frame_stack, num_envs, reward_type, reward_shaping, num_ball_to_win, max_steps,
                     finish_reward):
    base_env_fn = get_env_fn(program, reward_type, reward_shaping, num_ball_to_win, max_steps, finish_reward)

    if num_envs > 1:
        env = SubprocVecEnv([base_env_fn for _ in range(num_envs)])
    else:
        env = DummyVecEnv([base_env_fn])

    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    return env

def evaluate_ppo_policy(model, env, n_training_envs, n_eval_episodes=10, deterministic=True,
                        render=False, callback=None, reward_threshold=None,
                        return_episode_rewards=False):
    """
    Runs policy for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0

        zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)
        while not done:
            # concatenate obs
            # https://github.com/hill-a/stable-baselines/issues/166
            zero_completed_obs[0, :] = obs

            action, state = model.predict(zero_completed_obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step([action[0]])
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render(mode='human')
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: ' \
                                               '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

def evaluate_one(saved_file_name):
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    program.set_correct()

    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                           max_steps=1000, finish_reward=0)

    model = PPO2.load("./saved_models/train_graph_rad_cutout_reward_shape_False_6M.zip")

    mean_reward, std_reward = evaluate_ppo_policy(model, env, n_training_envs=8, n_eval_episodes=10)

    print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))


if __name__ == '__main__':
    print("==========cutout========")
    evaluate_one("./saved_models/train_graph_rad_cutout_reward_shape_False_6M.zip")
    print("==========cutout_color==========")
    evaluate_one("./saved_models/train_graph_rad_cutout_color_reward_shape_False_6M.zip")