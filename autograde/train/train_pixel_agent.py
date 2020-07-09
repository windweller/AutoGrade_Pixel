import os
import argparse
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.atari_wrappers import ScaledFloatFrame, WarpFrame

from stable_baselines.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, StopTrainingOnRewardThreshold
from stable_baselines.common.policies import CnnLstmPolicy
from stable_baselines.common.evaluation import evaluate_policy

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame

os.environ['SDL_VIDEODRIVER'] = 'dummy'

# A very good note indeed!
# For recurrent policies, with PPO2, the number of environments run in parallel
# should be a multiple of nminibatches.

# not worrying about seed right now...

def get_env_fn(program, reward_type, reward_shaping):
    # we add all necessary wrapper here
    def make_env():
        env = BouncePixelEnv(program, reward_type, reward_shaping)
        # env = WarpFrame(env)
        # env = ScaledFloatFrame(env)
        env = ResizeFrame(env)
        # env = ScaledFloatFrame(env)
        return env
    return make_env


def make_general_env(program, frame_stack, num_envs, reward_type, reward_shaping, seed=0):
    base_env_fn = get_env_fn(program, reward_type, reward_shaping)

    if num_envs > 1:
        env = SubprocVecEnv([base_env_fn for _ in range(num_envs)])
    else:
        env = DummyVecEnv([base_env_fn])

    if frame_stack > 1:
        env = VecFrameStack(env, n_stack=frame_stack)

    return env

def main():
    program = Program()
    program.set_correct()

    # env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # pylint: disable=E1101

    # TODO: Next step plan: Add Monitor with episode reward and episode length!!!

    with tf.Session(config=config):

        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./saved_models/self_minus_finish_reward_and_shaping/",
                                                 name_prefix="ppo2_cnn_lstm_default")

        # TODO: we can go back to nminibatches=1 training...
        model = PPO2(CnnLstmPolicy, env, n_steps=256, learning_rate=5e-4, gamma=0.99,
                     verbose=1, nminibatches=4, tensorboard_log="./tensorboard_self_minus_finish_and_shaping_log/")

        # model.learn(total_timesteps=1000 * 5000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')
        model.learn(total_timesteps=1000 * 10000, callback=CallbackList([checkpoint_callback]), tb_log_name='PPO2')

        model.save("./saved_models/ppo2_cnn_lstm_better_reward_and_shaping_final")

        # single_env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
        # recurrent policy, no stacking!
        single_env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False)
        # AssertionError: You must pass only one environment when using this function
        # But then, the NN is expecting shape of (8, ...)
        mean_reward, std_reward = evaluate_policy(model, single_env, n_eval_episodes=10)
        print("final model mean reward {}, std reward {}".format(mean_reward, std_reward))

        env.close()

def test_observations():
    # ========= Execute some random actions and observe ======
    program = Program()
    program.set_correct()
    env = make_general_env(program, 1, 1, ONLY_SELF_SCORE, False)

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

if __name__ == '__main__':
    # main()

    # ====== Some rough testing code =====
    # program = Program()
    # program.set_correct()
    # env = make_general_env(program, 4, 1, ONLY_SELF_SCORE)
    # obs = env.reset()
    # (1, 84, 84, 4)
    # FRAME_STACK=4

    # env = make_general_env(program, 4, 2, ONLY_SELF_SCORE)
    # obs = env.reset()
    # (2, 84, 84, 4)

    # This should be correct!

    # env = BouncePixelEnv(program, ONLY_SELF_SCORE)
    # obs = env.reset()
    # (400, 400, 3)

    # env = get_env_fn(program, ONLY_SELF_SCORE)()
    # obs = env.reset()
    # # (84, 84, 1)
    #
    # print(obs.shape)
    # env.close()

    test_observations()
