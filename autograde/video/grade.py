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

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, help="")
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    evaluate_on_rewards_and_values(args.n, args.skip)

