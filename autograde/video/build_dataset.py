# This is the first attempt (draft) at quickly building a video training dataset
# We need the following things:
# 1. Need to be able to select programs (two ways of loading in -- a line-by-line JSON strings

# There might be a parallelization problem

import random
import json
import time
import glob
from os.path import join as pjoin
import pathlib

import cv2
import numpy as np
from tqdm import tqdm

from stable_baselines import PPO2
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecVideoRecorder
from stable_baselines.bench import Monitor

import gym
from gym.wrappers.monitoring.video_recorder import ImageEncoder

from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO
from autograde.rl_envs.wrappers import ResizeFrame
from autograde.train.train_pixel_agent import make_general_env

import os

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dsp'

class GenerateVideoToy(object):
    # This is specifically designed for testing purposes
    def __init__(self, correct_program_folder, broken_program_folder, model_file, n_train_env, save_dir,
                 no_split=False):
        """
        :param n_train_env: this is set up during training time. PPO setting is 8.
        """
        broken_program_files = os.listdir(broken_program_folder)
        programs = {}
        for p in broken_program_files:
            program = json.load(open(pjoin(broken_program_folder, p)))
            programs[p.rstrip(".json")] = program

        self.broken_programs = programs

        random.seed(1234)
        filenames = list(self.broken_programs.keys())
        random.shuffle(filenames)
        # self.train_file_names = filenames[:5]  # half!
        # self.test_file_names = filenames[5:]

        self.broken_train_file_names = filenames
        self.test_file_names = []

        correct_program_files = os.listdir(correct_program_folder)
        # correct_program = json.load(open(pjoin(correct_program_folder, correct_program_file)))
        corr_programs = {}
        for p in correct_program_files:
            program = json.load(open(pjoin(correct_program_folder, p)))
            corr_programs[p.rstrip(".json")] = program
        self.corr_programs = corr_programs
        corr_filenames = list(self.corr_programs.keys())
        self.correct_train_file_names = corr_filenames

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

        for _ in tqdm(range(num_videos)):
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

    def modified_model_predict(self, observation, state=None, mask=None, deterministic=True):

        if state is None:
            state = self.model.initial_state
        if mask is None:
            mask = [False for _ in range(self.model.n_envs)]

        observation = np.array(observation)
        vectorized_env = BaseRLModel._is_vectorized_observation(observation, self.model.observation_space)

        observation = observation.reshape((-1,) + self.model.observation_space.shape)
        actions, values, states, _ = self.model.step(observation, state, mask, deterministic=deterministic)

        # <bound method LstmPolicy.step of <stable_baselines.common.policies.CnnLstmPolicy object at 0x1a31ee4198>>
        # actions, values, self.states, neglogpacs = self.model.step(obs, state, self.dones)

        # action, next_state = self.model.predict(obs, state=state, deterministic=True)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states, values

    def play_program(self, program_json, num_evals=1):
        # we only collect rewards and other stats in this method, no recording!

        program = Program()
        program.loads(program_json)

        env = make_general_env(program, 1, 1, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                                   max_steps=500,
                                   finish_reward=100)


        all_episode_rewards = []
        all_episode_values = []
        all_episode_dones = []

        for _ in tqdm(range(num_evals)):

            obs = env.reset()
            done, state = False, None
            step_reward = []
            step_value = []
            step_done = []

            zero_completed_obs = np.zeros((self.n_env,) + env.observation_space.shape)

            for _ in tqdm(range(500)):
                # in this recording scheme, if an environment is terminated/done, it's automatically
                # reset
                zero_completed_obs[0, :] = obs

                # action, state = self.model.predict(zero_completed_obs, state=state, deterministic=True)
                action, state, values_flat = self.modified_model_predict(zero_completed_obs, state=state, deterministic=True)
                obs, reward, done, _info = env.step(action)
                step_reward.append(reward[0])
                step_value.append(values_flat[0])
                step_done.append(done[0])

            # all_episode_rewards.append(step_reward)

        # return np.mean(all_episode_rewards), all_episode_rewards
        return np.sum(step_reward), step_reward, step_value, step_done

    def generate_training(self, num=20, enlarge_ratio=5, pos_label=None, neg_label=None):
        # This is for the AnomalyDataset class
        # where you save into a positive and negative folder
        # save into mp4

        pos_label = 'pos_train' if pos_label is None else pos_label
        neg_label = 'neg_train' if neg_label is None else neg_label

        for f_n in self.correct_train_file_names:
            p = self.corr_programs[f_n]
            self.record_program(p, f_n, pos_label, num * enlarge_ratio)

        for f_n in self.broken_train_file_names:
            p = self.broken_programs[f_n]
            self.record_program(p, f_n, neg_label, num)

        print("training video generation finished")

    def generate_test(self, num=20):
        for f_n in self.test_file_names:
            p = self.broken_programs[f_n]
            self.record_program(p, f_n, "external_neg_test", num)

        print("testing video generation finished")

class BounceVideoSplitter(object):
    # this is created to split parallel recordings of videos to individual mp4
    # currently the dimension is specifically designed for Bounce
    @staticmethod
    def split_all_videos_from(root_dir, n_train_env, max_frames=1500):
        # the grid is 3x3, from Bounce
        # and the last block is black/empty since we only have 8 environments
        mp4_files = glob.glob(root_dir + "/*.mp4")

        frame_shape = (400, 400, 3)
        frames_per_sec = 50
        import time
        start = time.time()

        for mp4_file_loc in tqdm(mp4_files):
            cap = cv2.VideoCapture(mp4_file_loc)

            outs = [ImageEncoder(mp4_file_loc.replace(".mp4", "-split-{}.mp4".format(i)), frame_shape, frames_per_sec,
                                 frames_per_sec) for i in range(n_train_env)]

            while (cap.isOpened()):
                ret, frame = cap.read()

                if ret == True:
                    height = 400
                    width = 400

                    imgwidth, imgheight = frame.shape[0], frame.shape[1]
                    rows = np.int(imgheight / height)
                    cols = np.int(imgwidth / width)

                    cnt = 0
                    for i in range(rows):
                        for j in range(cols):
                            # print (i,j)
                            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                            if cnt < 8:
                                # ffmpeg uses RGB color, CV2 uses BGR
                                transposed_color_frame = cv2.cvtColor(frame[box[0]:box[2], box[1]:box[3], :],
                                                                      cv2.COLOR_BGR2RGB)
                                outs[cnt].capture_frame(transposed_color_frame)
                            cnt += 1
                else:
                    break

            [o.close() for o in outs]
            cap.release()

        # print("Used {} secs".format(time.time() - start))

    @staticmethod
    def crop(im, height, width):
        # we are not call this method...in order to speed up
        # im = Image.open(infile)
        imgwidth, imgheight = im.shape[0], im.shape[1]
        rows = np.int(imgheight / height)
        cols = np.int(imgwidth / width)
        for i in range(rows):
            for j in range(cols):
                # print (i,j)
                box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                yield im[box[0]:box[2], box[1]:box[3], :]


class RLController(object):
    # This supports parallel
    # We are just generating videos, no concern for anything else
    # Definitely no "splitting"

    def __init__(self, model_file, n_train_env):
        """
        :param n_train_env: this is set up during training time. PPO setting is 8.
        """
        self.n_train_env = n_train_env

        self.model_file = model_file
        self.model = None

        self.n_env = n_train_env

    def load_programs(self, folder):
        program_files = os.listdir(folder)
        programs = {}
        for p in program_files:
            program = json.load(open(pjoin(folder, p)))
            programs[p.rstrip(".json")] = program

        return programs

    def load_model(self):
        self.model = PPO2.load(self.model_file)


    def record_program(self, program_json, program_name, save_dir, program_label='pos_videos', num_videos=8,
                       save_reward=False):
        # program_label={'pos_train', 'neg_train', 'pos_test', 'neg_test'}

        assert num_videos % self.n_train_env == 0, "We parallelize the environment, " \
                                                   "num_videos must be divisible by {}".format(self.n_train_env)

        program = Program()
        program.loads(program_json)

        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=3,
                               max_steps=1500, finish_reward=100)

        env = VecVideoRecorder(env, "{}/{}".format(save_dir, program_label),
                               record_video_trigger=lambda x: x == 0, video_length=1500,
                               name_prefix="ppo2-{}".format(program_name))

        # We will collect the reward and then store in a text file
        # it will be saved as a comma separated numbers

        record_rounds = num_videos // self.n_train_env
        all_episode_rewards = []

        for _ in tqdm(range(record_rounds)):
            obs = env.reset()
            done, state = False, None
            episode_reward = np.zeros(self.n_train_env)

            for _ in tqdm(range(1500)):
                # in this recording scheme, if an environment is terminated/done, it's automatically
                # reset
                action, state = self.model.predict(obs, state=state, deterministic=True)
                obs, reward, done, _info = env.step(action)
                episode_reward += reward

            all_episode_rewards.extend(episode_reward.tolist())

        if save_reward:
            rew_str = ",".join([str(r) for r in all_episode_rewards]) + '\n'
            f = open(pjoin("{}/{}".format(save_dir, program_label), "ppo2-{}-rewards.txt".format(program_name)), 'w')
            f.write(rew_str)
            f.close()

    def modified_model_predict(self, observation, state=None, mask=None, deterministic=True):

        if state is None:
            state = self.model.initial_state
        if mask is None:
            mask = [False for _ in range(self.model.n_envs)]

        observation = np.array(observation)
        vectorized_env = BaseRLModel._is_vectorized_observation(observation, self.model.observation_space)

        observation = observation.reshape((-1,) + self.model.observation_space.shape)
        actions, values, states, _ = self.model.step(observation, state, mask, deterministic=deterministic)

        # <bound method LstmPolicy.step of <stable_baselines.common.policies.CnnLstmPolicy object at 0x1a31ee4198>>
        # actions, values, self.states, neglogpacs = self.model.step(obs, state, self.dones)

        # action, next_state = self.model.predict(obs, state=state, deterministic=True)

        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.model.action_space, gym.spaces.Box):
            clipped_actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states, values

    def play_program(self, program_json, num_evals=8, return_stats=False):
        # we only collect rewards and other stats in this method, no recording!
        assert num_evals % self.n_train_env == 0

        program = Program()
        program.loads(program_json)

        # originally it was num_ball_to_win=3, max_steps=1500, finish_reward=? might be 100
        env = make_general_env(program, 1, 8, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=3,
                                   max_steps=1500,
                                   finish_reward=100)  # [-130, 160]

        record_rounds = num_evals // self.n_train_env

        all_episode_rewards = []
        all_episode_step_rewards, all_episode_step_values, all_episode_step_dones = [], [], []

        for _ in tqdm(range(record_rounds)):

            obs = env.reset()
            done, state = False, None
            episode_reward = np.zeros(self.n_train_env)
            step_rewards, step_values, step_dones = [], [], []

            for _ in tqdm(range(1000)):
                # in this recording scheme, if an environment is terminated/done, it's automatically
                # reset
                # action, state = self.model.predict(obs, state=state, deterministic=True)
                action, state, value = self.modified_model_predict(obs, state=state, deterministic=True)
                obs, reward, done, _info = env.step(action)
                episode_reward += reward
                if return_stats:
                    step_rewards.append(reward.tolist())
                    step_values.append(value.tolist())
                    step_dones.append(done.tolist())

            all_episode_rewards.extend(episode_reward.tolist())

            if return_stats:
                all_episode_step_rewards.append(step_rewards)
                all_episode_step_values.append(step_values)
                all_episode_step_dones.append(step_dones)

        if return_stats:
            return np.mean(all_episode_rewards), all_episode_rewards, all_episode_step_rewards, \
               all_episode_step_values, all_episode_step_dones
        else:
            return np.mean(all_episode_rewards), all_episode_rewards

    def generate_videos(self, correct_program_folder, broken_program_folder,
                        save_dir,
                        num=8, enlarge_ratio=5, pos_label=None, neg_label=None):
        # This is for the AnomalyDataset class
        # where you save into a positive and negative folder
        # save into mp4

        self.broken_programs = self.load_programs(broken_program_folder)
        self.broken_train_file_names = list(self.broken_programs.keys())

        self.corr_programs = self.load_programs(correct_program_folder)
        self.correct_train_file_names = list(self.corr_programs.keys())

        start = time.time()

        pos_label = 'pos_train' if pos_label is None else pos_label
        neg_label = 'neg_train' if neg_label is None else neg_label

        for f_n in tqdm(self.correct_train_file_names):
            p = self.corr_programs[f_n]
            self.record_program(p, f_n, save_dir, pos_label, num * enlarge_ratio)

        for f_n in tqdm(self.broken_train_file_names):
            p = self.broken_programs[f_n]
            self.record_program(p, f_n, save_dir, neg_label, num)

        print("training video generation finished, took {} secs".format(time.time() - start))

def generate_toy_program_videos():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/broken_small")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/correct_small")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    gvp = GenerateVideoToy(corr_folder, folder, model_file, n_train_env=8, save_dir="./rec_vidoes_toy_programs_n20/")
    gvp.load_model()

    gvp.generate_training(20)
    gvp.generate_test(20)

def generate_top10_program_videos():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/top_10_broken")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/correct_small")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    gvp = GenerateVideoToy(corr_folder, folder, model_file, n_train_env=8, save_dir="./rec_vidoes_top_10_programs_n20_480_mixed_theme/")
    gvp.load_model()

    gvp.generate_training(50, enlarge_ratio=6)

def generate_n100_unique_program_videos():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/100_uniq_programs/broken")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/100_uniq_programs/correct")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    gvp = GenerateVideoToy(corr_folder, folder, model_file, n_train_env=8,
                           save_dir="./rec_vidoes_100_uniq_programs_n10_1000/")
    gvp.load_model()

    gvp.generate_training(10, enlarge_ratio=1, pos_label='pos_eval', neg_label='neg_eval')

def parallel_gen_test():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/broken_small")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/correct_small")
    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    rlc = RLController(model_file, n_train_env=8)

    rlc.load_model()

    import time
    start = time.time()
    rlc.record_program(pjoin(corr_folder, "correct_sample.json"), "correct", "./test_parallel_record/", 'pos_videos', 8, save_reward=True)

    BounceVideoSplitter.split_all_videos_from("./test_parallel_record/pos_videos/", 8)

    print(time.time() - start, "secs")

def parallel_gen_100_unique_program_videos():
    broken_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/100_uniq_programs/broken")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/100_uniq_programs/correct")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")

    rlc = RLController(model_file, n_train_env=8)

    rlc.load_model()
    import time
    start = time.time()

    rlc.generate_videos(broken_folder, corr_folder, "./rec_vidoes_100_uniq_programs_n10_800/", 8, enlarge_ratio=1,
                        pos_label='pos_eval', neg_label='neg_eval')

    print(time.time() - start, "secs")

def split_100_unique_program_videos():
    import time
    start = time.time()

    BounceVideoSplitter.split_all_videos_from("./rec_vidoes_100_uniq_programs_n10_800/pos_eval/", 8)
    BounceVideoSplitter.split_all_videos_from("./rec_vidoes_100_uniq_programs_n10_800/neg_eval/", 8)

    print(time.time() - start, "secs")

def video_split_test():
    cap = cv2.VideoCapture('./test_parallel_record/ppo2-correct-step-0-to-step-2000.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # outs = [cv2.VideoWriter('./test_parallel_record/output-{}.mp4'.format(i), fourcc, 50.0, (400, 400))
    #         for i in range(8)]

    frame_shape = (400, 400, 3)
    frames_per_sec = 50

    outs = [ImageEncoder('./test_parallel_record/output-{}.mp4'.format(i), frame_shape, frames_per_sec,
                           frames_per_sec) for i in range(8)]

    while (cap.isOpened()):
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret == True:
            # write the flipped frame
            # a quick cut!
            height = 400
            width = 400

            imgwidth, imgheight = frame.shape[0], frame.shape[1]
            rows = np.int(imgheight / height)
            cols = np.int(imgwidth / width)

            cnt=0
            for i in range(rows):
                for j in range(cols):
                    # print (i,j)
                    box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
                    if cnt < 8:
                        # outs[cnt].write(frame[box[0]:box[2], box[1]:box[3], :])
                        transposed_color_frame = cv2.cvtColor(frame[box[0]:box[2], box[1]:box[3], :], cv2.COLOR_BGR2RGB)
                        outs[cnt].capture_frame(transposed_color_frame)
                    cnt += 1

            # cv2.imshow('frame', frame)
            # if 0xFF == ord('q'):
            #     break
        else:
            break

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    [o.close() for o in outs]
    cap.release()
    cv2.destroyAllWindows()

def play_to_grade_test():
    folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/broken_small")
    corr_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs/correct_small")

    model_file = pjoin(pathlib.Path(__file__).parent.parent.absolute(),
                       "train/saved_models/bounce_ppo2_cnn_lstm_one_ball_mixed_theme/ppo2_cnn_lstm_default_mixed_theme_final.zip")
    gvp = GenerateVideoToy(corr_folder, folder, model_file, n_train_env=8, save_dir="./rec_vidoes_toy_programs_n20/")
    gvp.load_model()

    program_folder = pjoin(pathlib.Path(__file__).parent.parent.absolute(), "envs/bounce_programs")

    avg_reward, step_reward, step_value, step_done = gvp.play_program(pjoin(program_folder, "correct_small", "correct_sample.json"), 1)

    print(avg_reward)
    for reward, value, done in zip(step_reward, step_value, step_done):
        print(reward, value, done)


def play_to_grade_test_parallel():
    pass

if __name__ == '__main__':

    # generate_toy_program_videos()
    # generate_top10_program_videos()

    # generate_n100_unique_program_videos()

    # parallel_gen_test()

    # took 10094.358183145523 secs = 2.8h
    # parallel_gen_100_unique_program_videos()

    # took 4811.676005601883 secs = 1.3h
    # split_100_unique_program_videos()

    # video_split_test()

    # play_to_grade_test()

    pass
