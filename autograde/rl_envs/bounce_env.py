import os
from os.path import join as pjoin

import PIL
import PIL.Image

import gym
from gym import spaces
from gym.wrappers.monitoring.video_recorder import ImageEncoder

import pygame
import numpy as np

from autograde.envs.bounce import Bounce, Program, screen_height, screen_width, \
    WHEN_LEFT_ARROW, WHEN_RIGHT_ARROW, fps

ONLY_SELF_SCORE = "only_self_score"
SELF_MINUS_HALF_OPPO = "self_minus_half_oppo"


def define_action_space(action_set):
    return spaces.Discrete(len(action_set))


def define_observation_space(screen_height, screen_width):
    return spaces.Box(low=0, high=255,
                      shape=(screen_height, screen_width, 3), dtype=np.uint8)


def define_physical_observation_space(shape=(3,)):
    # (x, y, direction)
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float)


class BounceRecording(gym.Env):
    def __init__(self, program: Program, recording_dir=None):
        self.program = program

        self.bounce = Bounce(program)
        self.recording_dir = recording_dir
        os.makedirs(recording_dir, exist_ok=True)
        self.recorded_frames = []

    def get_image(self):
        """
        https://mail.python.org/pipermail/python-list/2006-August/371647.html
        https://pillow.readthedocs.io/en/stable/reference/Image.html
        :return:
        """
        image_str = pygame.image.tostring(self.bounce.screen, 'RGB')
        image = PIL.Image.frombytes(mode='RGB', size=(screen_height, screen_width), data=image_str)
        image_np_array = np.array(image)
        return image_np_array

    def run(self):

        self.bounce.when_run_execute()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    # we save here
                    if self.recording_dir is not None:
                        rec_id = np.random.randint(0, 100000)
                        np.savez_compressed(open(pjoin(self.recording_dir, "game_frames_{}.npz".format(rec_id)), 'wb'),
                                            frames=np.array(self.recorded_frames, dtype=np.uint8))

                        return pjoin(self.recording_dir, "game_frames_{}.npz".format(rec_id))

            self.bounce.paddle.stop_moving()  # always stop paddle running at first

            keys = pygame.key.get_pressed()

            if self.bounce.when_left_arrow(keys):
                cmds = self.bounce.cc.remove_bounce(self.program[WHEN_LEFT_ARROW])
                list(map(lambda c: self.bounce.engine.execute(c), cmds))

            if self.bounce.when_right_arrow(keys):
                cmds = self.bounce.cc.remove_bounce(self.program[WHEN_RIGHT_ARROW])
                list(map(lambda c: self.bounce.engine.execute(c), cmds))

            # after a series of update...
            self.bounce.bg.draw(self.bounce.screen)
            self.bounce.score_board.draw(self.bounce.screen)

            # draw PyMunk objects (don't need this when actually playing)
            # app.space.debug_draw(draw_options)

            self.bounce.sync_sprite()
            self.bounce.all_sprites.draw(self.bounce.screen)

            self.bounce.clock.tick(fps)
            self.bounce.space.step(1 / fps)

            pygame.display.flip()

            if self.recording_dir is not None:
                self.recorded_frames.append(self.get_image())


class BouncePixelEnv(gym.Env):
    """
    We define game over in here
    game is over when player scores 5 points
    or opponent scores 5 points
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, program: Program, reward_type):

        assert reward_type in {ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO}
        self.reward_type = reward_type

        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.num_envs = 1
        self.viewer = None

        self.program = program

        self.bounce = Bounce(program)
        self.action_space = define_action_space(self.bounce.action_set)
        self.observation_space = define_observation_space(screen_height, screen_width)

    def step(self, action):
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action >= len(self.bounce.action_set):
            raise Exception("Can't choose an action based on index {}".format(action))

        prev_score = self.bounce.score_board.own
        prev_oppo_score = self.bounce.score_board.opponent

        # map action into key
        action_key = self.bounce.action_set[action]

        self.bounce.act(action_key)

        score = self.bounce.score_board.own
        oppo_score = self.bounce.score_board.opponent

        score_diff = score - prev_score
        oppo_score_diff = oppo_score - prev_oppo_score

        done = False
        if score == 5 or oppo_score == 5:
            done = True

        # considering "discount", reward should be higher
        # we can add reward shaping if necessary (through wrappers)
        if self.reward_type == SELF_MINUS_HALF_OPPO:
            reward = (score_diff - oppo_score_diff) * 20
            reward = max(-10, reward)
            # -10 vs. 20, we care more about goal than to catch the ball
            assert reward in {20, -10, 0}  # highest reward is 100
        else:
            reward = score_diff * 20  # we only care about sending the ball in; each ball in is 1 point
            # full points: 100

        # TODO: add reward shaping if training fails, to a wrapper, but should give out object information
        return self.get_image(), reward, done, {"score": score, "oppo_score": oppo_score}

    def reset(self):
        # we take the shortcut -- re-create the instance
        self.bounce = Bounce(self.program)
        return self.get_image()

    def render(self, mode='human'):
        img = self.get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def render_obs(self, obs):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(obs)
        return self.viewer.isopen

    def close(self):
        return

    def get_image(self):
        """
        https://mail.python.org/pipermail/python-list/2006-August/371647.html
        https://pillow.readthedocs.io/en/stable/reference/Image.html
        :return:
        """
        image_str = pygame.image.tostring(self.bounce.screen, 'RGB')
        image = PIL.Image.frombytes(mode='RGB', size=(screen_height, screen_width), data=image_str)
        image_np_array = np.array(image)
        return image_np_array


def convert_np_to_video(frames_name, output_video_name):
    assert '.mp4' in output_video_name, "The format is mp4"

    frames = np.load(frames_name)['frames']
    frame_shape = (400, 400, 3)
    frames_per_sec = 50
    ime = ImageEncoder(output_video_name, frame_shape, frames_per_sec)

    for f in frames:
        ime.capture_frame(f)

    ime.close()


if __name__ == '__main__':
    # use "numpy_to_mp4.py" to turn numpy to actual mp4

    program = Program()
    program.set_correct()
    # program.set_correct_with_theme()
    # program.load("../envs/bounce_programs/demo1.json")
    app = Bounce(program)
    app.run()

    import sys
    sys.exit(0)

    # record videos, and then convert them!
    program = Program()
    # program.load("../envs/bounce_programs/demo1.json")
    program.load("../envs/bounce_programs/demo2.json")
    app = BounceRecording(program, "./bounce_gameplay_recordings/")
    filename = app.run()

    # filename = "./bounce_gameplay_recordings/game_frames_46008.npz"

    convert_np_to_video(filename, "./bounce_gameplay_recordings/video2.mp4")
