import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecVideoRecorder

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO

try:
    from . import train_pixel_agent
except:
    import train_pixel_agent


def evaluate():
    # evaluate on the same environment
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    program.set_correct()

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=1, max_steps=1000,
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
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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
    model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    program = Program()
    if setting == 'hardcourt':
        program.set_correct()
    elif setting == 'retro':
        program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=1, max_steps=3000,
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
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.set_correct_with_theme()

    n_training_envs = 8  # originally training environments
    n_eval_episodes = 50
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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


def evaluate_change_theme_hard():
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
    {"when run": ["launch new ball"],
      "when left arrow": ["move left"],
      "when right arrow": ["move right"],
      "when ball hits paddle": ["bounce ball", "set scene 'random'"],
      "when ball hits wall": ["bounce ball", "set scene 'random'", "set ball 'random'", "set paddle 'random'"],
      "when ball in goal": ["score point", "launch new ball"],
      "when ball misses paddle": ["score opponent point",
                                  "launch new ball"]}
    """)

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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


def record_change_theme_hard():
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
    {"when run": ["launch new ball"],
      "when left arrow": ["move left"],
      "when right arrow": ["move right"],
      "when ball hits paddle": ["bounce ball", "set scene 'random'"],
      "when ball hits wall": ["bounce ball", "set scene 'random'", "set ball 'random'", "set paddle 'random'"],
      "when ball in goal": ["score point", "launch new ball"],
      "when ball misses paddle": ["score opponent point",
                                  "launch new ball"]}
    """)

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
                                             finish_reward=100)

    env = VecVideoRecorder(env, "./rec_videos/",
                           record_video_trigger=lambda x: x == 0, video_length=3000,
                           name_prefix="ppo2-cnn-lstm-mixed-theme-final-agent-{}")

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

def evaluate_random_speed_change():
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
        {"when run": ["launch new ball", "set 'random' ball speed", "set 'random' paddle speed"],
          "when left arrow": ["move left"],
          "when right arrow": ["move right"],
          "when ball hits paddle": ["bounce ball"],
          "when ball hits wall": ["bounce ball"],
          "when ball in goal": ["score point", "launch new ball"],
          "when ball misses paddle": ["score opponent point",
                                      "launch new ball"]}
        """)

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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

# This is known to fail
def evaluate_slow_paddle_fast_ball():
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
        {"when run": ["launch new ball", "set 'very fast' ball speed", "set 'very slow' paddle speed"],
          "when left arrow": ["move left"],
          "when right arrow": ["move right"],
          "when ball hits paddle": ["bounce ball"],
          "when ball hits wall": ["bounce ball"],
          "when ball in goal": ["score point", "launch new ball"],
          "when ball misses paddle": ["score opponent point",
                                      "launch new ball"]}
        """)

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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

# This should succeed
def evaluate_fast_paddle_fast_ball():
    model = PPO2.load(
        "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
        {"when run": ["launch new ball", "set 'very fast' ball speed", "set 'very fast' paddle speed"],
          "when left arrow": ["move left"],
          "when right arrow": ["move right"],
          "when ball hits paddle": ["bounce ball"],
          "when ball hits wall": ["bounce ball"],
          "when ball in goal": ["score point", "launch new ball"],
          "when ball misses paddle": ["score opponent point",
                                      "launch new ball"]}
        """)

    n_training_envs = 8  # originally training environments
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=5, max_steps=3000,
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

if __name__ == '__main__':
    pass
    # evaluate()
    # evaluate_five_ball()

    # evaluate_retro()
    # evaluate_retro_five_ball()

    # evaluate_change_theme_hard()

    # evaluate_random_speed_change()
    # evaluate_slow_paddle_fast_ball()
    # evaluate_fast_paddle_fast_ball()

    # record video
    # record_five_ball_video()
    # record_five_ball_video("retro")

    # record_change_theme_hard()

    # test_observations()
