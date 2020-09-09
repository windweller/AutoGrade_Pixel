"""
We generate grades here...
"""

import numpy as np

import torch

import os
import glob
import json
import time
from os.path import join as pjoin
import pathlib
from tqdm import tqdm

import wandb

from collections import defaultdict

from autograde.video.build_dataset import RLController

def evaluate_on_rewards_and_values(n, skip):
    # this is play-to-grade reward collector

    wandb.init(project="autograde-rollout", name="{}_uniq_programs_skip_{}".format(n, skip))

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    rlc = RLController(model_file, n_train_env=8)

    rlc.load_model()

    program_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs")

    save_stats_dir = './eval_reward_value_stats_n{}_skip{}/'.format(n, skip)
    os.makedirs(save_stats_dir, exist_ok=True)

    start = time.time()

    prog = 0
    for uniq_program_loc in tqdm(glob.glob(pjoin(program_folder, "{}_uniq_programs_skip_{}".format(n, skip), "correct", "*.json"))):
        avg_reward, rewards, step_rewards, step_values, step_dones = rlc.play_program(uniq_program_loc, 8,
                                                                                      return_stats=True)
        rew_str = ",".join([str(r) for r in rewards]) + '\n'
        f = open(save_stats_dir + 'correct_{}_rewards.txt'.format(uniq_program_loc.split('/')[-1].rstrip(".json")), 'w')
        f.write(rew_str)
        f.close()
        # save other stats
        filename = save_stats_dir + 'correct_{}_info.json'.format(uniq_program_loc.split('/')[-1].rstrip(".json"))
        f = open(filename, 'w')
        json.dump({'step_rewards': step_rewards,
                   'step_values': step_values,
                   'step_dones': step_dones}, f)
        f.close()

        prog += 1
        time_so_far = time.time() - start
        time_per_program = time_so_far / prog
        estimated_time_to_complete = time_per_program * ((n-skip) - prog)

        wandb.log({"Estimated Time to Complete (secs)":estimated_time_to_complete,
                   "Estimated Time to Complete (min)": estimated_time_to_complete / 60,
                   "Spent time (secs)": time_so_far})


    for uniq_program_loc in tqdm(glob.glob(pjoin(program_folder, "{}_uniq_programs_skip_{}".format(n, skip), "broken", "*.json"))):
        avg_reward, rewards, step_rewards, step_values, step_dones = rlc.play_program(uniq_program_loc, 8,
                                                                                      return_stats=True)
        rew_str = ",".join([str(r) for r in rewards]) + '\n'
        f = open(save_stats_dir + 'broken_{}_rewards.txt'.format(uniq_program_loc.split('/')[-1].rstrip(".json")), 'w')
        f.write(rew_str)
        f.close()
        # save other stats
        filename = save_stats_dir + 'broken_{}_info.json'.format(uniq_program_loc.split('/')[-1].rstrip(".json"))
        f = open(filename, 'w')
        json.dump({'step_rewards': step_rewards,
                   'step_values': step_values,
                   'step_dones': step_dones}, f)

        # shape: (1, 1000, 8) -- if put in Numpy
        f.close()

        prog += 1
        time_so_far = time.time() - start
        time_per_program = time_so_far / prog
        estimated_time_to_complete = time_per_program * ((n - skip) - prog)

        wandb.log({"Estimated Time to Complete (secs)": estimated_time_to_complete,
                   "Estimated Time to Complete (min)": estimated_time_to_complete / 60,
                   "Spent time (secs)": time_so_far})

    print("Time took {} secs".format(time.time() - start))

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

def run_evaluate_on_rewards_and_values():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    evaluate_on_rewards_and_values(args.n, args.skip)

def gen_traj_for_correct_program_rewards_and_values():
    # two things, one for themes (8 themes)
    # 25 speed settings

    # save each one.
    # this is a "parallel" situation...so we won't use this to construct Table 1 or 2.

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    rlc = RLController(model_file, n_train_env=8)
    rlc.load_model()

    save_stats_dir = './reference_eval_reward_value_stats_correct_programs_8_theme_24_speed/'
    os.makedirs(save_stats_dir, exist_ok=True)

    # each program we run 8 times (24 + 8) = 32
    # one speed "very slow paddle" "very fast ball" does not work (so 24 speed programs, not 25)
    # to just tip things more in our favor

    start = time.time()
    pbar = tqdm(total=32)

    # speed

    choices = ['very slow', 'slow', 'normal', 'fast', 'very fast']
    for ball_speed in choices:
        for paddle_speed in choices:
            setting_name = "ball_{}_paddle_{}".format(ball_speed.replace(" ", '_'), paddle_speed.replace(" ", '_'))
            program_json = setup_speed_json_string(ball_speed, paddle_speed)

            avg_reward, rewards, step_rewards, step_values, step_dones = rlc.play_program(program_json, 8,
                                                                                          return_stats=True)

            rew_str = ",".join([str(r) for r in rewards]) + '\n'
            f = open(
                save_stats_dir + 'correct_speed_{}_rewards.txt'.format(setting_name),
                'w')
            f.write(rew_str)
            f.close()
            # save other stats
            filename = save_stats_dir + 'correct_speed_{}_info.json'.format(setting_name)
            f = open(filename, 'w')
            json.dump({'step_rewards': step_rewards,
                       'step_values': step_values,
                       'step_dones': step_dones}, f)

            # shape: (1, 1000, 8) -- if put in Numpy
            f.close()
            pbar.update(1)

    # theme
    paddle_opts = ['hardcourt', 'retro']
    ball_opts = ['hardcourt', 'retro']
    background_opts = ['hardcourt', 'retro']

    for bg in background_opts:
        for pt in paddle_opts:
            for bt in ball_opts:
                setting_name = "{}_{}_{}".format(bg, pt, bt)
                program_json = setup_theme_json_string(bg, bt, pt)
                avg_reward, rewards, step_rewards, step_values, step_dones = rlc.play_program(program_json, 8,
                                                                                              return_stats=True)

                rew_str = ",".join([str(r) for r in rewards]) + '\n'
                f = open(
                    save_stats_dir + 'correct_speed_{}_rewards.txt'.format(setting_name),
                    'w')
                f.write(rew_str)
                f.close()
                # save other stats
                filename = save_stats_dir + 'correct_speed_{}_info.json'.format(setting_name)
                f = open(filename, 'w')
                json.dump({'step_rewards': step_rewards,
                           'step_values': step_values,
                           'step_dones': step_dones}, f)

                # shape: (1, 1000, 8) -- if put in Numpy
                f.close()
                pbar.update(1)

    print("Totally took {} secs".format(time.time() - start))

def gen_traj_for_reference_broken_program_rewards_and_values():
    # 10 broken programs
    # so need to run 8 * 3 times...to create a balanced dataset

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    rlc = RLController(model_file, n_train_env=8)
    rlc.load_model()

    save_stats_dir = './reference_eval_reward_value_stats_broken_10_programs/'
    os.makedirs(save_stats_dir, exist_ok=True)

    pbar = tqdm(total=10)

    program_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/broken_small/")

    for uniq_program_loc in glob.glob(pjoin(program_folder, "*.json")):
        avg_reward, rewards, step_rewards, step_values, step_dones = rlc.play_program(uniq_program_loc, 8*3,
                                                                                      return_stats=True)
        rew_str = ",".join([str(r) for r in rewards]) + '\n'
        f = open(save_stats_dir + 'broken_{}_rewards.txt'.format(uniq_program_loc.split('/')[-1].rstrip(".json")), 'w')
        f.write(rew_str)
        f.close()
        # save other stats
        filename = save_stats_dir + 'broken_{}_info.json'.format(uniq_program_loc.split('/')[-1].rstrip(".json"))
        f = open(filename, 'w')
        json.dump({'step_rewards': step_rewards,
                   'step_values': step_values,
                   'step_dones': step_dones}, f)
        f.close()
        pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    run_evaluate_on_rewards_and_values()
    # gen_traj_for_correct_program_rewards_and_values()
    # gen_traj_for_reference_broken_program_rewards_and_values()