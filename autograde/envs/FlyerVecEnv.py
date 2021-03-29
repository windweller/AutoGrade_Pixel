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

FLYER_RES_H = 400
FLYER_RES_W = 400
FLYER_NUM_DISCRETE_ACTIONS = 16

KEYMAP = ["", "ArrowLeft", "ArrowUp", "ArrowLeft|ArrowUp",
          "ArrowRight", "ArrowLeft|ArrowRight",
          "ArrowUp|ArrowRight", "ArrowLeft|ArrowUp|ArrowRight",
          "ArrowDown", "ArrowLeft|ArrowDown",
          "ArrowUp|ArrowDown", "ArrowLeft|ArrowUp|ArrowDown",
          "ArrowRight|ArrowDown", "ArrowLeft|ArrowRight|ArrowDown",
          "ArrowUp|ArrowRight|ArrowDown", "ArrowLeft|ArrowUp|ArrowRight|ArrowDown"]

headers = {"Content-Type": "application/json"}

class FlyerVecEnv(VecEnv):

    def __init__(self, num_envs=1, js_filename="flyer1", pixel=True, frames=2):

        # group number of observations together
        assert frames > 1
        
        self.num_envs = num_envs
        self.pixel    = pixel

        self.buf_rew  = np.zeros([num_envs], dtype=np.float32)
        self.buf_done = np.zeros([num_envs], dtype=np.bool)
        self.buf_info = [{} for _ in range(num_envs)]
        self.score    = np.zeros([num_envs], dtype=np.float32)
        self.coin_pos = np.zeros([num_envs, 2], dtype=np.float32)
        self.closed   = False


        action_space = gym.spaces.Discrete(FLYER_NUM_DISCRETE_ACTIONS)

        init_data = {"code":    js_filename,
                     "process": num_envs,
                     "format":  "img|state" if pixel else "state",
                     "frames":  frames}

        response = requests.post("http://localhost:3300/init/1",
                                  headers=headers,
                                  data=json.dumps(init_data))



        if pixel:
            self.buf_state = np.zeros([num_envs, FLYER_RES_H, FLYER_RES_W, 3],
                                      dtype=np.uint8)
            obs_space = gym.spaces.Box(low=0, high=255,
                                       shape=[FLYER_RES_H, FLYER_RES_W, 3],
                                       dtype=np.uint8)
        else:
            num_objects = len(response.json()[0]["state"]["sprites"])
            self.buf_state = np.zeros([num_envs, num_objects, 2], dtype=np.float32)
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
                sprites = [(c["animationLabel"],c["x"],c["y"]) for c in state["state"]["sprites"]]
                self.buf_state[i] = np.array([[c[1], c[2]] for c in sorted(sprites)])

            if reset:
                self.buf_rew[i] = 0
                self.buf_done[i] = False
                self.score[i] = 0
                for c in state["state"]["sprites"]:
                    if c["animationLabel"] == "coin":
                        self.coin_pos[i] = [c["x"], c["y"]]
                continue

            self.buf_rew[i] = 0
            for c in state["state"]["sprites"]:
                if c["animationLabel"] == "coin":
                    new_pos = [c["x"], c["y"]]
                    if not np.allclose(self.coin_pos[i], new_pos, atol=1e-4):
                        self.coin_pos[i] = new_pos
                        self.buf_rew[i] = 1
                        self.score[i] += 1
                elif c["animationLabel"] == "fly_bot":
                    print(i, c["x"], c["y"])
            self.buf_info[i]["score"] = self.score[i]
            for c in state["state"]["texts"]:
                if c["text"] == "Game Over!":
                    self.buf_done[i] = True
                    break


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
        step_data = {"actions":[{"key": KEYMAP[a]} for a in actions]}
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
            plt.imshow(self.buf_state.reshape(-1, FLYER_RES_W, 3))
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
