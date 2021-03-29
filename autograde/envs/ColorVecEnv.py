import gym
import gym.spaces
from gym.envs.classic_control import rendering
from stable_baselines.common.vec_env import VecEnv

import numpy as np
import requests
import json
import os
import util
import matplotlib.pyplot as plt
import time

COLOR_RES_H = 450
COLOR_RES_W = 320
COLOR_NUM_DISCRETE_ACTIONS = 20

headers = {"Content-Type": "application/json"}

class ColorVecEnv(VecEnv):

    def __init__(self, num_envs=1, js_filename="color1", pixel=True,
                 mouse_action=True, scale_action=False):

        assert (mouse_action) or (not scale_action)
        # group number of observations together

        self.num_envs = num_envs
        self.pixel = pixel
        self.mouse_action = mouse_action
        self.scale_action = scale_action

        self.buf_rew  = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_info = [{} for _ in range(num_envs)]
        self.score1  = np.zeros([num_envs], dtype=np.float32)
        self.score2  = np.zeros([num_envs], dtype=np.float32)
        self.closed = False


        if self.mouse_action:
            action_space = gym.spaces.Discrete(COLOR_NUM_DISCRETE_ACTIONS)
        else:
            if self.scale_action:
                action_space = gym.spaces.Box(low=0, high=1, shape=[2],
                                              dtype=np.float32)
            else:
                action_space = gym.spaces.Box(low=[0,0],
                                              high=[COLOR_RES_H, COLOR_RES_W],
                                              dtype=np.float32)


        init_data = {"code":    js_filename,
                     "process": num_envs,
                     "format":  "img|state" if pixel else "state"}

        response = requests.post("http://localhost:3300/init/1",
                                  headers=headers,
                                  data=json.dumps(init_data))



        if pixel:
            self.buf_state = np.zeros([num_envs, COLOR_RES_H, COLOR_RES_W, 3],
                                      dtype=np.uint8)
            obs_space = gym.spaces.Box(low=0, high=255,
                                       shape=[COLOR_RES_H, COLOR_RES_W, 3],
                                       dtype=np.uint8)
        else:
            num_objects = sum(['backgroundColor' in c for c in response.json()[0]["state"]])
            self.buf_state = np.zeros([num_envs, num_objects, 4], dtype=np.float32)
            obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=[num_objects, 2],
                                       dtype=np.float32)

        super().__init__(
            num_envs=num_envs,
            observation_space=obs_space,
            action_space=action_space
        )

    def _apply_tick_func(self, new_states, envs, reset=False):
        for i in envs:
            if not reset and self.buf_done[i]:
                continue
            state = new_states[i]
            if self.pixel:
                self.buf_state[i] = util.jpg_b64_to_rgb(state["img"])
            else:
                components = [(c["id"], c["backgroundColor"])
                              for c in state["state"] if "backgroundColor" in c]
                self.buf_state[i] = np.array([util.getRGBA(c[1])
                                             for c in sorted(components)])

            curr_player = 0
            for c in state["state"]:
                if c["id"] == "player1_highlight" and util.getRGBA(c["backgroundColor"])[3] > 0.01:
                    curr_player = 2
                    break
                elif c["id"] == "player2_highlight" and util.getRGBA(c["backgroundColor"])[3] > 0.01:
                    curr_player = 1
                    break

            for c in state["state"]:
                if c["id"] == "score1_label":
                    s1 = int(c["text"])
                    if curr_player == 1:
                        self.buf_rew[i] = s1 - self.score1[i] if not reset else 0
                    self.buf_info[i]['player'] = curr_player
                    self.buf_info[i]['cum_score_1'] = s1
                    self.score1[i] = s1
                elif c["id"] == "score2_label":
                    s2 = int(c["text"])
                    if curr_player == 2:
                        self.buf_rew[i] = s2 - self.score2[i] if not reset else 0
                        self.buf_info[i]['player'] = curr_player
                    self.buf_info[i]['cum_score_2'] = s2
                    self.score2[i] = s2

    def reset(self, envs_to_reset=None):

        reset_data = {"actions": envs_to_reset} if envs_to_reset is not None else {}
        new_states = requests.post("http://localhost:3300/reset/1",
                                   headers=headers,
                                   data=json.dumps(reset_data)).json()
        if envs_to_reset is None:
            envs_to_reset = range(self.num_envs)
        self._apply_tick_func(new_states, envs_to_reset, reset=True)

        return self.buf_state

    def step_async(self, actions):
        if self.mouse_action:
            if self.scale_action:
                step_data = {"actions":\
                              [{"x": int(np.floor(action[0] * (COLOR_RES_W-1))),
                                "y": int(np.floor(action[1] * (COLOR_RES_H-1)))}
                               for action in actions]}
            else:
                step_data = {"actions":[{"x": int(np.floor(action[0])),
                                         "y": int(np.floor(action[1]))}
                                         for action in actions]}
        else:
            step_data = {"actions":[{"grid": int(a)} for a in actions]}

        new_states = requests.post("http://localhost:3300/step/1",
                                   headers=headers,
                                   data=json.dumps(step_data)).json()
        self._apply_tick_func(new_states, range(self.num_envs), reset=False)


    def step_wait(self):
        # return the buffer
        return self.buf_state, self.buf_rew, self.buf_done, self.buf_info

    def get_images(self):
        assert self.pixel
        return self.buf_state

    def close(self):
        requests.post("http://localhost:3300/close/1", headers=headers)
        self.closed = True

    def render(self, i=None):
        assert self.pixel
        if i is None:
            plt.imshow(self.buf_state.reshape(-1, COLOR_RES_W, 3))
        else:
            plt.imshow(self.buf_state[i])
        plt.show()

    def __del__(self):
        if not self.closed:
            self.close()

    def get_attr(self, attr_name, indices=None):
        pass

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def seed(self, s):
        pass
