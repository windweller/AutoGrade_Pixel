import os
import argparse
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.atari_wrappers import ScaledFloatFrame, WarpFrame
from stable_baselines.common.vec_env import VecEnv

import numpy as np
from gym.wrappers import TimeLimit

from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, \
    StopTrainingOnRewardThreshold
from stable_baselines.common.policies import CnnLstmPolicy

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame

os.environ['SDL_VIDEODRIVER'] = 'dummy'


# A very good note indeed!
# For recurrent policies, with PPO2, the number of environments run in parallel
# should be a multiple of nminibatches.

# not worrying about seed right now...

def get_env_fn(program, reward_type, reward_shaping, num_ball_to_win, max_steps=1000, finish_reward=0):
    # we add all necessary wrapper here
    def make_env():
        env = BouncePixelEnv(program, reward_type, reward_shaping, num_ball_to_win=num_ball_to_win,
                             finish_reward=finish_reward)
        # env = WarpFrame(env)
        # env = ScaledFloatFrame(env)
        env = ResizeFrame(env)
        # env = ScaledFloatFrame(env)
        env = TimeLimit(env,
                        max_episode_steps=max_steps)  # 20 seconds  # no skipping, it should be 3000. With skip, do 3000 / skip.
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
                env.render()
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


def train():
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    # TODO: if wrap monitor, we can get episodic reward

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path="./saved_models/self_minus_finish_reward_and_shaping/",
                                                 name_prefix="ppo2_cnn_lstm_default")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
        model = PPO2(CnnLstmPolicy, env, n_steps=256, learning_rate=5e-4, gamma=0.99,
                     verbose=1, nminibatches=4, tensorboard_log="./tensorboard_self_minus_finish_log/")

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=3000000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.save("./saved_models/ppo2_cnn_lstm_better_reward_and_shaping_final")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()

def train_random_mixed_theme():
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    # TODO: if wrap monitor, we can get episodic reward

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path="./saved_models/self_minus_finish_reward_mixed_theme/",
                                                 name_prefix="ppo2_cnn_lstm_default")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
        # not n_step=256, because shorter helps us learn a better policy; 128 = 2 seconds out
        # model = PPO2(CnnLstmPolicy, env, n_steps=256, learning_rate=5e-4, gamma=0.99,
        #              verbose=1, nminibatches=4, tensorboard_log="./tensorboard_self_minus_finish_reward_mixed_theme_log/")

        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=5000000 * 3, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')  # 10M

        model.save("./saved_models/self_minus_finish_reward_mixed_theme")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()


def test_observations():
    # ========= Execute some random actions and observe ======
    program = Program()
    program.set_correct()
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, False, num_ball_to_win=1, max_steps=1000,
                           finish_reward=0)

    import numpy as np
    from autograde.rl_envs.utils import SmartImageViewer

    viewer = SmartImageViewer()

    obs = env.reset()
    print(env.action_space.n)
    for i in range(1000):
        action = np.random.randint(env.action_space.n, size=1)
        obs, rewards, dones, info = env.step(action)
        # env.render()
        obs = obs.squeeze(0)
        viewer.imshow(obs)
        if rewards[0] != 0:
            print(rewards)

        if dones[0]:
            break

    print(i)
    env.close()


def evaluate():
    # evaluate on the same environment
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct()

    n_training_envs = 8  # originally training environments
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1, max_steps=1000,
                           finish_reward=0)

    obs = env.reset()
    done, state = False, None
    episode_reward = 0.0
    episode_length = 0

    zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)
    while not done:
        # concatenate obs
        # https://github.com/hill-a/stable-baselines/issues/166
        zero_completed_obs[0, :] = obs

        action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
        obs, reward, done, _info = env.step([action[0]])
        episode_reward += reward

        episode_length += 1
        env.render()


def evaluate_five_ball_with_render():
    # evaluate on a five-ball environment
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct()

    n_training_envs = 8  # originally training environments
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=5, max_steps=3000,
                           finish_reward=100)

    obs = env.reset()
    done, state = False, None
    episode_reward = 0.0
    episode_length = 0

    zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)

    while not done:
        # concatenate obs
        # https://github.com/hill-a/stable-baselines/issues/166
        zero_completed_obs[0, :] = obs

        action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
        obs, reward, done, _info = env.step([action[0]])
        episode_reward += reward

        episode_length += 1
        env.render()

    print("total reward: {}".format(episode_reward))
    print("episode length: {}".format(episode_length))


def record_five_ball_video(setting='hardcourt'):
    from stable_baselines.common.vec_env import VecVideoRecorder
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    if setting == 'hardcourt':
        program.set_correct()
    elif setting == 'retro':
        program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=5, max_steps=3000,
                           finish_reward=100)

    env = VecVideoRecorder(env, "./rec_videos/",
                           record_video_trigger=lambda x: x == 0, video_length=3000,
                           name_prefix="ppo2-cnn-lstm-final-agent-{}".format(setting))

    obs = env.reset()
    done, state = False, None
    episode_reward = 0.0
    episode_length = 0

    zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)

    while not done:
        # concatenate obs
        # https://github.com/hill-a/stable-baselines/issues/166
        zero_completed_obs[0, :] = obs

        action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
        obs, reward, done, _info = env.step([action[0]])
        episode_reward += reward

        episode_length += 1
        env.render()

    print("total reward: {}".format(episode_reward))
    print("episode length: {}".format(episode_length))


def evaluate_five_ball():
    # evaluate on a five-ball environment
    # model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_2000000_steps.zip")
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct()

    n_training_envs = 8  # originally training environments
    n_eval_episodes = 50
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=5, max_steps=3000,
                           finish_reward=100)

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

            action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
            obs, reward, done, _info = env.step([action[0]])
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("Average episode length: {}".format(np.mean(episode_lengths)))
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))


def evaluate_retro():
    # evaluate on the same environment
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=5, max_steps=3000,
                           finish_reward=0)

    obs = env.reset()
    done, state = False, None
    episode_reward = 0.0
    episode_length = 0

    zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)
    while not done:
        # concatenate obs
        # https://github.com/hill-a/stable-baselines/issues/166
        zero_completed_obs[0, :] = obs

        action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
        obs, reward, done, _info = env.step([action[0]])
        episode_reward += reward

        episode_length += 1
        env.render()


def evaluate_retro_five_ball():
    # evaluate on a five-ball environment
    # model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_2000000_steps.zip")
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    n_eval_episodes = 50
    env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=5, max_steps=3000,
                           finish_reward=100)

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

            action, state = model.predict(zero_completed_obs, state=state, deterministic=True)
            obs, reward, done, _info = env.step([action[0]])
            episode_reward += reward
            episode_length += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("Average episode length: {}".format(np.mean(episode_lengths)))
    print("Mean reward: {}, std: {}".format(mean_reward, std_reward))


if __name__ == '__main__':
    pass
    train_random_mixed_theme()
    # train()

    # evaluate()
    # evaluate_five_ball()

    # evaluate_retro()
    # evaluate_retro_five_ball()

    # record video
    # record_five_ball_video()
    # record_five_ball_video("retro")

    # test_observations()

