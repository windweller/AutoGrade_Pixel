# This is the first attempt (draft) at quickly building a video training dataset
# We need the following things:
# 1. Need to be able to select programs (two ways of loading in -- a line-by-line JSON strings

# There might be a parallelization problem

import random
import json
import os
from os.path import join as pjoin
import pathlib

import numpy as np

from stable_baselines import PPO2
from stable_baselines.common.vec_env import VecVideoRecorder

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame
from autograde.train.train_pixel_agent import make_general_env

class GenerateVideoToy(object):
    # This is specifically designed for testing purposes
    def __init__(self, correct_program_folder, broken_program_folder, model_file, n_train_env, save_dir):
        """
        :param n_train_env: this is set up during training time. PPO setting is 8.
        """
        broken_program_files = os.listdir(broken_program_folder)
        programs = {}
        for p in broken_program_files:
            program = json.load(open(pjoin(broken_program_folder, p)))
            programs[p.rstrip(".json")] = program

        self.programs = programs

        random.seed(1234)
        filenames = list(self.programs.keys())
        random.shuffle(filenames)
        self.train_file_names = filenames[:5]  # half!
        self.test_file_names = filenames[5:]

        correct_program_file = os.listdir(correct_program_folder)[0]
        correct_program = json.load(open(pjoin(correct_program_folder, correct_program_file)))
        self.correct_program = correct_program

        self.model_file = model_file
        self.model = None

        self.n_env = n_train_env
        self.root_save_dir = save_dir

    def load_model(self):
        self.model = PPO2.load(self.model_file)

    def record_program(self, program_json, program_name, program_label='pos_videos', num_videos=20):
        # program_label={'pos_train', 'neg_train', 'pos_test', 'neg_test'}

        program = Program()
        program.loads(program_json)

        # we should be able to do parellel recording!! WOW!!! Easy!
        env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=3,
                               max_steps=2000,
                               finish_reward=100)

        env = VecVideoRecorder(env, "{}/{}".format(self.root_save_dir, program_label),
                               record_video_trigger=lambda x: x == 0, video_length=2000,
                               name_prefix="ppo2-{}".format(program_name))

        for _ in range(num_videos):
            obs = env.reset()
            done, state = False, None
            episode_reward = 0.0
            episode_length = 0

            zero_completed_obs = np.zeros((self.n_env,) + env.observation_space.shape)

            while not done:
                # concatenate obs
                # https://github.com/hill-a/stable-baselines/issues/166
                zero_completed_obs[0, :] = obs

                action, state = self.model.predict(zero_completed_obs, state=state, deterministic=True)
                obs, reward, done, _info = env.step([action[0]])
                episode_reward += reward

                episode_length += 1
                # env.render()

    def generate_training(self, num=20):
        # This is for the AnomalyDataset class
        # where you save into a positive and negative folder
        # save into mp4
        self.record_program(self.correct_program, 'correct_sample', 'pos_train', num * 5)

        for f_n in self.train_file_names:
            p = self.programs[f_n]
            self.record_program(p, f_n, "neg_train", num)

        print("training video generation finished")

    def generate_test(self, num=20):
        for f_n in self.test_file_names:
            p = self.programs[f_n]
            self.record_program(p, f_n, "external_neg_test", num)

        print("testing video generation finished")

def generate_toy_program_videos():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/broken_small")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/correct_small")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    gvp = GenerateVideoToy(corr_folder, folder, model_file, n_train_env=8, save_dir="./rec_vidoes_toy_programs/")
    gvp.load_model()

    gvp.generate_training(20)
    gvp.generate_test(20)

if __name__ == '__main__':

    generate_toy_program_videos()

    pass