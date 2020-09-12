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

os.environ['SDL_VIDEODRIVER'] = 'dummy'

def build_random_impala_cnn(scaled_image, depths=[16,32,32], **conv_kwargs):
    """
    unscaled_images

    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """

    layer_num = 0
    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value

        out = tf.nn.relu(inputs)

        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    # out = tf.cast(unscaled_images, tf.float32) / 255.

    # In RAD baselines, all images are unscaled, so they scale it
    # However, in Stable baselines, all images are scaled? (at least that's for nature_cnn)

    # but since this is random CNN...we are going to scale this!!

    out = tf.cast(scaled_image, tf.float32) / 255.

    out = tf.layers.conv2d(out, 3, 3,
                           padding='same',
                           kernel_initializer=tf.initializers.glorot_normal(),
                           trainable=False,
                           name='randcnn')
    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out

class RandomCnnLstmPolicy(LstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        super(RandomCnnLstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                                            layer_norm=False, feature_extraction="cnn", cnn_extractor=build_random_impala_cnn, **_kwargs)


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


# _policy_registry = {
#     ActorCriticPolicy: {
#         "CnnPolicy": CnnPolicy,
#         "CnnLstmPolicy": CnnLstmPolicy,
#         "CnnLnLstmPolicy": CnnLnLstmPolicy,
#         "MlpPolicy": MlpPolicy,
#         "MlpLstmPolicy": MlpLstmPolicy,
#         "MlpLnLstmPolicy": MlpLnLstmPolicy,
#     }
# }

def train():
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    hyperparams = {
        "finish_reward": 80, # 0,
        "reward_shaping": True, # False,
        "n_steps": 256, # 256,
        'learning_rate': 5e-4, # 5e-4,
        'max_steps': 1000,
        'policy_type': 'CnnLstmPolicy' # 'CnnLstmPolicy'
    }

    import wandb
    wandb.init(sync_tensorboard=True, project="autograde-bounce",
               name="self_minus_oppo_finish_reward_shaping_retrain4_n256",
               config=hyperparams)

    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    # TODO: if wrap monitor, we can get episodic reward

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path="./saved_models/self_minus_oppo_finish_reward_shaping_retrain4_n256/",
                                                 name_prefix="ppo2_cnn_lstm_retrain4")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=hyperparams['reward_shaping'],
                               num_ball_to_win=1,
                               max_steps=hyperparams['max_steps'], finish_reward=0)
        model = PPO2(hyperparams['policy_type'], env, n_steps=hyperparams['n_steps'],
                     learning_rate=hyperparams['learning_rate'], gamma=0.99,
                     verbose=1, nminibatches=4, tensorboard_log="./tensorboard_self_minus_oppo_finish_reward_shaping_retrain4_n256_log/")

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO,
                                      reward_shaping=hyperparams['reward_shaping'], num_ball_to_win=1,
                                      max_steps=hyperparams['max_steps'], finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=3000000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.save("./saved_models/ppo2_cnn_lstm_self_minus_oppo_finish_reward_shaping_retrain4_n256")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO,
                                      reward_shaping=hyperparams['reward_shaping'], num_ball_to_win=1,
                                      max_steps=hyperparams['max_steps'], finish_reward=0)
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
    # program.set_correct()
    program.load("./autograde/envs/bounce_programs/mixed_theme_train.json")

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

        model = PPO2.load("./saved_models/ppo2_cnn_lstm_default_final.zip")
        model.set_env(env)
        model.tensorboard_log = "./tensorboard_self_minus_finish_reward_mixed_theme_log/"
        model.verbose = 1
        model.nminibatches = 4
        model.learning_rate = 5e-4

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=5000000, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='PPO2')  # 10M

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

def train_rad(data_aug_name, reward_shaping=False):
    # instead of training w/ mixed theme, we train with RAD instead, and evaluate if it's any good

    import wandb
    wandb.init(sync_tensorboard=True, project="autograde-bounce",
               name="self_minus_oppo_rad_{}_reward_shape_{}".format(data_aug_name, reward_shaping))

    from rad_ppo2 import PPO2
    # The PPO was not modified, but Runner is.

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=25000,  # 500 * 8 = 4000 | 400000 | 200000
                                                 save_path="./saved_models/self_minus_oppo_rad_{}_reward_shape_{}/".format(data_aug_name, reward_shaping),
                                                 name_prefix="ppo2_cnn_lstm_rad_{}".format(data_aug_name))

        if not reward_shaping:
            env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
        else:
            env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=True, num_ball_to_win=1,
                                   max_steps=1000, finish_reward=0)  # before we did 0...

        # not n_step=256, because shorter helps us learn a better policy; 128 = 2 seconds out
        # model = PPO2(CnnLstmPolicy, env, n_steps=256, learning_rate=5e-4, gamma=0.99,
        #              verbose=1, nminibatches=4, tensorboard_log="./tensorboard_self_minus_finish_reward_mixed_theme_log/")

        model = PPO2.load("./saved_models/ppo2_cnn_lstm_default_final.zip")
        # PPO2 set data_aug
        model.data_aug = data_aug_name # 'cutout_color'
        model.set_env(env)

        model.tensorboard_log = "./tensorboard_self_minus_oppo_rad_{}_reward_shaping_{}_log/".format(data_aug_name, reward_shaping)
        model.verbose = 1
        model.nminibatches = 4
        model.learning_rate = 5e-4

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=3000000, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='PPO2')  # 10M

        model.save("./saved_models/self_minus_oppo_rad_{}_reward_shaping_{}".format(data_aug_name, reward_shaping))

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        program.set_correct_retro_theme()
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)

        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()

def train_randomnet():
    # train randomnet with reward shaping
    # train for 6M steps (3M + 3M)

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    hyperparams = {
        "finish_reward": 0,  # 0,
        "reward_shaping": True,  # False,
        "n_steps": 256,  # 256,
        'learning_rate': 5e-4,  # 5e-4,
        'max_steps': 1000,
    }

    import wandb
    wandb.init(sync_tensorboard=True, project="autograde-bounce",
               name="self_minus_oppo_reward_shaping_train_randomnet",
               config=hyperparams)

    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    # TODO: if wrap monitor, we can get episodic reward

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path="./saved_models/self_minus_oppo_reward_shaping_randomnet_impala/",
                                                 name_prefix="ppo2_cnn_lstm_randomnet")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=hyperparams['reward_shaping'],
                               num_ball_to_win=1,
                               max_steps=hyperparams['max_steps'], finish_reward=0)

        model = PPO2(RandomCnnLstmPolicy, env, n_steps=hyperparams['n_steps'],
                     learning_rate=hyperparams['learning_rate'], gamma=0.99,
                     verbose=1, nminibatches=4,
                     tensorboard_log="./tensorboard_self_minus_oppo_reward_shaping_randomnet_impala_log/")

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO,
                                      reward_shaping=hyperparams['reward_shaping'], num_ball_to_win=1,
                                      max_steps=hyperparams['max_steps'], finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=6000000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.save("./saved_models/ppo2_cnn_lstm_self_minus_oppo_reward_shaping_randomnet_impala")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO,
                                      reward_shaping=hyperparams['reward_shaping'], num_ball_to_win=1,
                                      max_steps=hyperparams['max_steps'], finish_reward=0)
        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()

def train_random_ball_setting():
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    # program.set_correct()
    program.load("./autograde/envs/bounce_programs/mixed_theme_train.json")


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

        model = PPO2.load("./saved_models/ppo2_cnn_lstm_default_final.zip")
        model.set_env(env)
        model.tensorboard_log = "./tensorboard_self_minus_finish_reward_mixed_theme_log/"
        model.verbose = 1
        model.nminibatches = 4
        model.learning_rate = 5e-4

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=5000000 * 3, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='PPO2')  # 10M

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

def run_train():
    import argparse
    # train standard or mixed-theme start...for the training graph
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", action="store_true")
    args = parser.parse_args()
    # standard mixed-theme vs. curriculum mixed-theme
    # 6M vs. 3M + 3M

    import wandb
    name = 'curriculum' if args.curriculum else "standard"
    wandb.init(sync_tensorboard=True, project="autograde-bounce",
               name="paper_train_graph_mixed_{}".format(name))

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    # program.set_correct()
    program.load("./autograde/envs/bounce_programs/mixed_theme_train.json")

    # then model here we decide
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=250000,
                                                 save_path="./saved_models/paper_train_graph_mixed_{}/".format(name),
                                                 name_prefix="ppo2_cnn_lstm_default")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
        # not n_step=256, because shorter helps us learn a better policy; 128 = 2 seconds out

        if args.curriculum:
            model = PPO2.load("./saved_models/ppo2_cnn_lstm_default_final.zip")
            model.set_env(env)
            model.tensorboard_log = "./tensorboard_paper_train_graph_mixed_curriculum_3M_log/"
            model.verbose = 1
            model.nminibatches = 4
            model.learning_rate = 5e-4

            steps = 3000000
            save_name = 'curriculum'
        else:
            model = PPO2('CnnLstmPolicy', env, n_steps=256, learning_rate=5e-4, gamma=0.99,
                         verbose=1, nminibatches=4, tensorboard_log="./tensorboard_paper_train_graph_mixed_standard_6M_log/")
            steps = 6000000
            save_name = 'standard'

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.learn(total_timesteps=steps, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='PPO2')  # 3M

        model.save("./saved_models/paper_train_graph_mixed_{}".format(save_name))

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        program.set_correct_retro_theme()
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)

        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()

def train_speed_mixed():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddle_speed", type=str, help="slow or very_slow")
    args = parser.parse_args()

    import wandb
    name = 'slow' if args.paddle_speed else "very_slow"
    wandb.init(sync_tensorboard=True, project="autograde-bounce",
               name="paper_mixed_theme_continue_mixed_ball_speed_paddle_{}".format(name))

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    program = Program()
    # program.set_correct()
    if args.paddle_speed == 'slow':
        program.load("./autograde/envs/bounce_programs/correct_small/correct_sample_theme_with_slow_speed.json")
    else:
        program.load("./autograde/envs/bounce_programs/correct_small/correct_sample_with_theme_very_slow_speed.json")

    # then model here we decide
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    with tf.Session(config=config):
        checkpoint_callback = CheckpointCallback(save_freq=100000,
                                                 save_path="./saved_models/paper_mixed_theme_continue_mixed_ball_speed_paddle_{}/".format(args.paddle_speed),
                                                 name_prefix="ppo2_cnn_lstm_default")

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)

        model = PPO2.load("./saved_models/paper_train_graph_mixed_standard.zip")
        steps = 3000000

        # Eval first to make sure we can eval this...(otherwise there's no point in training...)
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)

        print("initial model mean reward {}, std reward {}".format(mean_reward, std_reward))

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.learn(total_timesteps=steps, callback=CallbackList([checkpoint_callback]),
                    tb_log_name='PPO2')  # 3M

        model.save("./saved_models/paper_mixed_theme_continue_mixed_ball_speed_paddle_{}".format(args.paddle_speed))

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        program.set_correct_retro_theme()
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                      max_steps=1000, finish_reward=0)

        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_ppo_policy(model, single_env, n_training_envs=8, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()


if __name__ == '__main__':
    pass
    # train_random_mixed_theme()
    # train()

    # train_rad('cutout_color')
    # train_rad('color_jitter')

    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_aug", type=str, help="")
    # parser.add_argument("--reward_shaping", action="store_true")
    # args = parser.parse_args()
    #
    # train_rad(args.data_aug, args.reward_shaping)

    # train_randomnet()

    # generate paper RL training graph
    # run_train()
    train_speed_mixed()