import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecVideoRecorder

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO

try:
    from . import train_pixel_agent
except:
    import train_pixel_agent

from tqdm import tqdm
import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dsp'

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
        program.set_correct_retro_theme()

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


def evaluate_five_ball(standard_model=True):
    # evaluate on a five-ball environment
    # model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_2000000_steps.zip")
    if standard_model:
        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    else:
        model = PPO2.load(
            "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.set_correct()

    n_training_envs = 8  # originally training environments
    n_eval_episodes = 10
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


def evaluate_retro(standard_model=True):
    # evaluate on the same environment
    if standard_model:
        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    else:
        model = PPO2.load(
            "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.set_correct_retro_theme()

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


def evaluate_retro_five_ball(standard_model=True):
    # evaluate on a five-ball environment
    # model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_2000000_steps.zip")
    if standard_model:
        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    else:
        model = PPO2.load(
            "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.set_correct_retro_theme()

    n_training_envs = 8  # originally training environments
    n_eval_episodes = 10
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


def evaluate_change_theme_hard_five_ball(standard_model=True):
    if standard_model:
        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    else:
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
    n_eval_episodes = 10

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

def evaluate_speed_five_ball(ball_speed, paddle_speed, standard_model=True):
    if standard_model:
        model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    else:
        model = PPO2.load(
            "./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    program = Program()
    program.loads("""
        {"when run": ["launch new ball", "set '"""+ball_speed+"""' ball speed", "set '"""+paddle_speed+"""' paddle speed"],
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
                                             num_ball_to_win=5, max_stteps=3000,
                                             finish_reward=100)
    n_eval_episodes = 5

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

def generate_a_few_speed_table():
    pass
    # print("Standard performance on hardcourt:")
    # evaluate_five_ball(True)
    # print("Mixed performance on hardcourt:")
    # evaluate_five_ball(False)
    # print("Standard performance on retro")
    # evaluate_retro_five_ball(True)
    # print("Mixed performance on retro:")
    # evaluate_retro_five_ball(False)
    # print("Standard performance on mixed:")
    # evaluate_change_theme_hard_five_ball(True)
    # print("Mixed performance on mixed:")
    # evaluate_change_theme_hard_five_ball(False)

    print("=====Fast ball, Fast paddle=====")
    print("Standard:")
    evaluate_speed_five_ball("fast", "fast", True)
    print("Mixed:")
    evaluate_speed_five_ball("fast", "fast", False)
    print("=====Fast ball, normal paddle=====")
    print("Standard:")
    evaluate_speed_five_ball("fast", "normal", True)
    print("Mixed:")
    evaluate_speed_five_ball("fast", "normal", False)
    print("=====Fast ball, slow paddle=====")
    print("Standard:")
    evaluate_speed_five_ball("fast", "slow", True)
    print("Mixed:")
    evaluate_speed_five_ball("fast", "slow", False)


# We evaluate RAD
import scipy

def mean_confidence_interval(data, confidence=0.9):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h, h

def evaluate_thematic_generalization(model, model_name, setting, n_eval_episodes=10):
    # This is 3-ball evaluation
    assert setting in {'mixed', 'retro', 'hardcourt'}

    program = Program()
    if setting == 'retro':
        program.set_correct_retro_theme()
    elif setting == 'hardcourt':
        program.set_correct()
    elif setting == 'mixed':
        program.loads("""
        {"when run": ["launch new ball", "set scene 'random'", "set ball 'random'", "set paddle 'random'"],
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
                                             num_ball_to_win=5, max_steps=2000,
                                             finish_reward=100)  # [-150, +200]

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

    mean, low, high, h = mean_confidence_interval(episode_rewards)

    print("{} Performance under theme {}".format(model_name, setting))
    print("Average episode length: {}".format(np.mean(episode_lengths)))
    print("Mean reward: {}, CI: {}-{}, range: {}".format(mean, low[0], high[0], h[0]))

def evaluate_rl_models_on_themes():
    # standard_model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    #
    # evaluate_thematic_generalization(standard_model, 'Standard Model', "hardcourt")
    # evaluate_thematic_generalization(standard_model, 'Standard Model', "retro")
    # evaluate_thematic_generalization(standard_model, 'Standard Model', "mixed")

    # curriculum_model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    #
    # evaluate_thematic_generalization(curriculum_model, 'Curriculum Model', "hardcourt")
    # evaluate_thematic_generalization(curriculum_model, 'Curriculum Model', "retro")
    # evaluate_thematic_generalization(curriculum_model, 'Curriculum Model', "mixed")

    # rad_model = PPO2.load("./saved_models/self_minus_oppo_rad_cutout_color.zip")
    #
    # evaluate_thematic_generalization(rad_model, 'RAD Cutout Color Model', "hardcourt")
    # evaluate_thematic_generalization(rad_model, 'RAD Cutout Color Model', "retro")
    # evaluate_thematic_generalization(rad_model, 'RAD Cutout Color Model', "mixed")

    # rad_model = PPO2.load("./saved_models/self_minus_oppo_rad_color_jitter.zip")
    #
    # evaluate_thematic_generalization(rad_model, 'RAD Color Jitter Model', "hardcourt")
    # evaluate_thematic_generalization(rad_model, 'RAD Color Jitter Model', "retro")
    # evaluate_thematic_generalization(rad_model, 'RAD Color Jitter Model', "mixed")

    pass

def get_performance(model, json_obj, n_eval_episodes=10, finish_reward=100, num_ball_to_win=5):
    # This is 3-ball evaluation

    program = Program()
    program.loads(json_obj)

    n_training_envs = 8  # originally training environments

    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=num_ball_to_win, max_steps=2000,
                                             finish_reward=finish_reward)  # [-150, +200] 20 * 5 + 100 = 200

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

    mean, low, high, h = mean_confidence_interval(episode_rewards)

    # print("{} Performance under theme {}".format(model_name, setting))
    # print("Average episode length: {}".format(np.mean(episode_lengths)))
    # print("Mean reward: {}, CI: {}-{}, range: {}".format(mean, low[0], high[0], h[0]))
    return mean, h[0]


def setup_theme_json_string(scene, ball, paddle):
    from string import Template

    random_program_str = Template("""
            {"when run": ["launch new ball", "set '${scene}' scene", "set '${ball}' ball", "set '${paddle}' paddle"],
              "when left arrow": ["move left"],
              "when right arrow": ["move right"],
              "when ball hits paddle": ["bounce ball"],
              "when ball hits wall": ["bounce ball"],
              "when ball in goal": ["score point", "launch new ball"],
              "when ball misses paddle": ["score opponent point",
                                          "launch new ball"]}
            """)

    return random_program_str.substitute(scene=scene, ball=ball, paddle=paddle)

def eval_one_model_on_variations(model, pbar):
    paddle_opts = ['hardcourt', 'retro']
    ball_opts = ['hardcourt', 'retro']
    background_opts = ['hardcourt', 'retro']

    header_row = []
    perf_row = []

    # 8 settings
    for bg in background_opts:
        for pt in paddle_opts:
            for bt in ball_opts:
                setting_name = "{}-{}-{}".format(bg, pt, bt)
                header_row.append(setting_name)
                program_json = setup_theme_json_string(bg, bt, pt)
                mean, r = get_performance(model, program_json, 10)
                perf_row.append("{:.1f} $\pm$ {:.1f}".format(mean, r))
                pbar.update(1)

    return header_row, perf_row

def generate_result_table1():
    # result table will be made by 5 balls, evaluated 10 times (single environment)
    # pbar = tqdm(total=8)
    pbar = tqdm(total=8 * 6)

    import csv

    file = open("./theme_invariance_model_eval.csv", 'w')
    writer = csv.writer(file)

    standard_model = PPO2.load("./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    header, perf = eval_one_model_on_variations(standard_model, pbar)

    print("Training Strategy," + ",".join(header))

    writer.writerow(["Training Strategy"] + header)

    print("Standard, " + ",".join(perf))

    writer.writerow(["Standard"] + perf)

    curriculum_model = PPO2.load(
        "./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    _, perf = eval_one_model_on_variations(curriculum_model, pbar)
    print("Curriculum, " + ",".join(perf))

    writer.writerow(["Curriculum"] + perf)

    colorjitter_model = PPO2.load("saved_models/self_minus_oppo_rad_color_jitter.zip")
    _, perf = eval_one_model_on_variations(colorjitter_model, pbar)
    print("Standard + color-jitter, " + ",".join(perf))

    writer.writerow(["Standard + color-jitter"] + perf)

    gray_model = PPO2.load("saved_models/self_minus_oppo_rad_gray.zip")
    _, perf = eval_one_model_on_variations(gray_model, pbar)
    print("Standard + gray-scale, " + ",".join(perf))

    writer.writerow(["Standard + gray-scale"] + perf)

    cutout_model = PPO2.load("saved_models/self_minus_oppo_rad_cutout.zip")
    _, perf = eval_one_model_on_variations(cutout_model, pbar)
    print("Standard + cutout, " + ",".join(perf))

    writer.writerow(["Standard + cutout"] + perf)

    cutout_color_model = PPO2.load("saved_models/self_minus_oppo_rad_cutout_color.zip")
    _, perf = eval_one_model_on_variations(cutout_color_model, pbar)
    print("Standard + cutout-color, " + ",".join(perf))

    writer.writerow(["Standard + cutout-color"] + perf)

    pbar.close()
    file.close()

def evaluate_one_model_on_table1():
    pbar = tqdm(total=8)

    standard_model = PPO2.load(
        "./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")

    header, perf = eval_one_model_on_variations(standard_model, pbar)
    print(header)
    print(perf)
    # TODO: evaluate the model!!!

def setup_speed_json_string(ball, paddle):
    from string import Template

    random_program_str = Template("""
            {"when run": ["launch new ball", "set '${ball}' ball speed", "set '${paddle}' paddle speed"],
              "when left arrow": ["move left"],
              "when right arrow": ["move right"],
              "when ball hits paddle": ["bounce ball"],
              "when ball hits wall": ["bounce ball"],
              "when ball in goal": ["score point", "launch new ball"],
              "when ball misses paddle": ["score opponent point",
                                          "launch new ball"]}
            """)

    return random_program_str.substitute(ball=ball, paddle=paddle)

def generate_result_table2():
    # speed variants!!
    # while you generate, you should save traj them too.
    pbar = tqdm(total=25)

    # curriculum_model = PPO2.load(
    #     "./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    # model = PPO2.load("./saved_models/paper_mixed_theme_continue_mixed_ball_speed_paddle_slow.zip")
    model = PPO2.load("./saved_models/paper_mixed_theme_continue_mixed_ball_speed_paddle_very_slow.zip")

    rows = []

    choices = ['very slow', 'slow', 'normal', 'fast', 'very fast']

    rows.append([' '] + choices) # header

    for paddle_speed in choices:
        # this "row" correspond to paddle speed as row (on left), ball speed as column (on top)
        row = []
        for ball_speed in choices:
            program_json = setup_speed_json_string(ball_speed, paddle_speed)
            mean, r = get_performance(model, program_json, 10)
            row.append("{:.1f} $\pm$ {:.1f}".format(mean, r))
            pbar.update(1)

        rows.append([paddle_speed] + row)

    import csv
    # file = open("./speed_invariance_curriculum_model_eval.csv", 'w')
    # file = open("./speed_invariance_mixed_theme_continue_mixed_ball_speed_paddle_slow_eval.csv", 'w')
    file = open("./speed_invariance_paper_mixed_theme_continue_mixed_ball_speed_paddle_very_slow.csv", 'w')
    writer = csv.writer(file)

    for row in rows:
        writer.writerow(row)

    file.close()

def investigate():
    standard_model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    program_json = setup_theme_json_string('hardcourt', 'hardcourt', 'hardcourt')

    program = Program()
    program.loads(program_json)

    n_training_envs = 8  # originally training environments

    # 5 balls, 2000 steps are necessary!! Can't be 1500
    # env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
    #                                          num_ball_to_win=5, max_steps=2000,
    #                                          finish_reward=100)  # [-150, +200] 20 * 5 + 100 = 200

    # see if 1000 and 3 balls is a good pairing...
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=3, max_steps=1500,
                                             finish_reward=100)  # [-130, +160] 20 * 3 + 100 = 160

    episode_rewards, episode_lengths = [], []
    num_balls_in = []
    for _ in range(1):

        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        num_ball = 0

        zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)
        while not done:
            # concatenate obs
            # https://github.com/hill-a/stable-baselines/issues/166
            zero_completed_obs[0, :] = obs

            action, state = standard_model.predict(zero_completed_obs, state=state, deterministic=True)
            obs, reward, done, _info = env.step([action[0]])
            episode_reward += reward
            episode_length += 1
            if reward > 0:
                num_ball += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean, low, high, h = mean_confidence_interval(episode_rewards)
    print(episode_reward)
    print(episode_length)
    print(num_ball)

def investigate2():
    standard_model = PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip")
    program_json = setup_speed_json_string('very slow', 'very slow')

    program = Program()
    program.loads(program_json)

    n_training_envs = 8  # originally training environments

    # see if 1000 and 3 balls is a good pairing...
    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=3, max_steps=1500,
                                             finish_reward=100)  # [-130, +160] 20 * 3 + 100 = 160

    episode_rewards, episode_lengths = [], []
    num_balls_in = []
    for _ in range(1):

        obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        num_ball = 0

        zero_completed_obs = np.zeros((n_training_envs,) + env.observation_space.shape)
        while not done:
            # concatenate obs
            # https://github.com/hill-a/stable-baselines/issues/166
            zero_completed_obs[0, :] = obs

            action, state = standard_model.predict(zero_completed_obs, state=state, deterministic=True)
            obs, reward, done, _info = env.step([action[0]])
            episode_reward += reward
            episode_length += 1
            if reward > 0:
                num_ball += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean, low, high, h = mean_confidence_interval(episode_rewards)
    print(episode_reward)
    print(episode_length)
    print(num_ball)


def get_full_performance(model, json_obj, n_eval_episodes=5, finish_reward=50, num_ball_to_win=5):
    # This is 3-ball evaluation

    program = Program()
    program.loads(json_obj)

    n_training_envs = 8  # originally training environments

    env = train_pixel_agent.make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False,
                                             num_ball_to_win=num_ball_to_win, max_steps=2000,
                                             finish_reward=finish_reward)  # [-150, +200] 20 * 5 + 100 = 200

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
            episode_reward += reward.item(0)
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    # mean, low, high, h = mean_confidence_interval(episode_rewards)

    # print("{} Performance under theme {}".format(model_name, setting))
    # print("Average episode length: {}".format(np.mean(episode_lengths)))
    # print("Mean reward: {}, CI: {}-{}, range: {}".format(mean, low[0], high[0], h[0]))
    return episode_rewards, episode_lengths

def eval_one_model_on_correct_programs(model, pbar, programs, save_dir):
    from os.path import join as pjoin
    import json

    for name, program_json in programs:
        try:
            episode_rewards, episode_lengths = get_full_performance(model, program_json, 10, 0, 5)
            print(episode_rewards)
            print(episode_lengths)
            json.dump({'episode_rewards': episode_rewards,
                    'episode_lengths':episode_lengths}, open(pjoin(save_dir, name), 'w'))
        except:
            pass
        pbar.update(1)

def get_model(model_name):
    # dic = {
    #     'Standard': PPO2.load("./saved_models/bounce_ppo2_cnn_lstm_one_ball/ppo2_cnn_lstm_default_final.zip"),
    #     'Mixed_Theme': PPO2.load("./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip"),
    #     "RAD_Cutout": PPO2.load("saved_models/self_minus_oppo_rad_cutout.zip"),
    #     "RAD_Cutout_Color": PPO2.load("saved_models/self_minus_oppo_rad_cutout_color.zip"),
    #     "RAD_Color_Jitter": PPO2.load("saved_models/self_minus_oppo_rad_color_jitter.zip"),
    #     "RAD_Gray_Scale": PPO2.load("saved_models/self_minus_oppo_rad_gray.zip")
    # }

    if model_name == 'Standard':
        return PPO2.load("./saved_models/ppo2_cnn_lstm_default_final.zip")
    elif model_name == "Mixed_Theme":
        # return PPO2.load(
        #     "./saved_models/paper_train_graph_mixed_standard.zip")
        return PPO2.load("./autograde/train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    elif model_name == "RAD_Cutout":
        return PPO2.load("saved_models/self_minus_oppo_rad_cutout.zip")
    elif model_name == "RAD_Cutout_Color":
        return PPO2.load("saved_models/self_minus_oppo_rad_cutout_color.zip")
    elif model_name == "RAD_Color_Jitter":
        return PPO2.load("saved_models/self_minus_oppo_rad_color_jitter.zip")
    elif model_name == "RAD_Gray_Scale":
        return PPO2.load("saved_models/self_minus_oppo_rad_gray.zip")

def evaluate_on_correct_programs():

    from os.path import join as pjoin
    import json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", type=str, help="")
    parser.add_argument("--model", type=str, help="")
    args = parser.parse_args()

    # we could use wandb to track progress
    # let's wait for now and see how fast/slow it is

    # load programs
    root_dir = "/home/aimingnie/AutoGrade/autograde/envs/bounce_programs/"
    program_dir = pjoin(root_dir, args.eval_folder)

    program_jsons = []
    for f_n in os.listdir(program_dir):
        program_json = json.load(open(pjoin(program_dir, f_n)))
        program_jsons.append((f_n, program_json))

    # remove 2 bad ones for one model
    # program_jsons = program_jsons[:7] + program_jsons[11:]

    pbar = tqdm(total=len(program_jsons))

    save_dir = "/home/aimingnie/AutoGrade/{}_result/{}".format(args.eval_folder, args.model)
    os.makedirs(save_dir, exist_ok=True)

    model = get_model(args.model)
    eval_one_model_on_correct_programs(model, pbar, program_jsons, save_dir)


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

    # generate_a_few_speed_table()

    # evaluate_rl_models_on_themes()
    # evaluate_rl_models_on_themes()

    # generate_result_table1()
    # investigate()

    # generate_result_table2()

    # investigate2()

    evaluate_on_correct_programs()
