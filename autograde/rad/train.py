import tensorflow as tf
import ppo2
from baselines.common.models import build_impala_cnn, cnn_lstm
from baselines.common.mpi_util import setup_mpi_gpus
from baselines.common.vec_env import (
    VecMonitor,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse

from baselines.common.vec_env import VecFrameStack, DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame
from gym.wrappers import TimeLimit

import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dsp'

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
        env = VecFrameStack(env, nstack=frame_stack)

    return env

def main():

    learning_rate = 5e-4
    ent_coef = .01
    gamma = .99
    lam = .95
    nsteps = 256
    nminibatches = 4 # 8
    ppo_epochs = 3
    clip_range = .2
    timesteps_per_proc = 20_000_000 # 200_000_000: hard 25_000_000: easy
    use_vf_clipping = True
    LOG_DIR = './log/'

    parser = argparse.ArgumentParser(description='Process procgen training arguments.')
    parser.add_argument('--env_name', type=str, default='bounce')
    parser.add_argument('--test_worker_interval', type=int, default=0)
    parser.add_argument('--num_envs', type=int, default=8)
    parser.add_argument('--data_aug', type=str, default='normal')
    parser.add_argument('--exp_name', type=str, default='try1')
    parser.add_argument('--log_filename', type=str, default='vec_monitor_log.csv')

    args = parser.parse_args()

    num_envs = args.num_envs

    program = Program()
    program.set_correct()

    test_worker_interval = args.test_worker_interval

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False
    
    #if args.num_levels < 50:
    #    timesteps_per_proc = 5_000_000
    
    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout'] if log_comm.Get_rank() == 0 else []
    # LOG_DIR += args.env_name + '/nlev_' +  str(args.num_levels) + '_mode_'
    # LOG_DIR += args.distribution_mode +'/' + args.data_aug + '/' + args.exp_name

    LOG_DIR += args.env_name + '/' + args.exp_name

    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")
    venv = make_general_env(program, 1, num_envs, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
    # venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=num_levels, start_level=args.start_level, distribution_mode=args.distribution_mode)
    # venv = VecExtractDictObs(venv, "rgb")

    venv = VecMonitor(
        venv=venv, filename=args.log_filename, keep_buf=100,
    )

    # venv = VecNormalize(venv=venv, ob=False)
    
    # eval env, unlimited levels
    # eval_venv = ProcgenEnv(num_envs=num_envs, env_name=args.env_name, num_levels=0,
    #                        start_level=args.test_start_level, distribution_mode=args.distribution_mode)
    # eval_venv = VecExtractDictObs(eval_venv, "rgb")

    eval_venv = make_general_env(program, 1, num_envs, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                               max_steps=1000, finish_reward=0)
    eval_venv = VecMonitor(
        venv=eval_venv, filename=None, keep_buf=100,
    )
    # eval_venv = VecNormalize(venv=eval_venv, ob=False)

    logger.info("creating tf session")
    setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    # conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2.learn(
        env=venv,
        eval_env=eval_venv,
        network='cnn_lstm',  # conv_fn
        total_timesteps=timesteps_per_proc,
        save_interval=50_000,  # 62,
        nsteps=nsteps,
        nminibatches=nminibatches,
        lam=lam,
        gamma=gamma,
        noptepochs=ppo_epochs,
        log_interval=1,
        ent_coef=ent_coef,
        mpi_rank_weight=mpi_rank_weight,
        # clip_vf=use_vf_clipping,
        comm=comm,
        lr=learning_rate,
        cliprange=clip_range,
        update_fn=None,
        init_fn=None,
        vf_coef=0.5,
        max_grad_norm=0.5,
        data_aug=args.data_aug,
    )

if __name__ == '__main__':
    main()