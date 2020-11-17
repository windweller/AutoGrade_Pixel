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
    WHEN_LEFT_ARROW, WHEN_RIGHT_ARROW, fps, PAUSE

ONLY_SELF_SCORE = "only_self_score"
SELF_MINUS_HALF_OPPO = "self_minus_half_oppo"


def define_action_space(action_set):
    return spaces.Discrete(len(action_set))


def define_observation_space(screen_height, screen_width):
    return spaces.Box(low=0, high=255,
                      shape=(screen_height, screen_width, 3), dtype=np.uint8)


def define_object_observation_space(shape):
    # shape = (3,)
    # (x, y, direction)
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float)

class PyGamePixelEnv(object):
    def __init__(self, game):
        self.game = game

    def get_image(self):
        """
        https://mail.python.org/pipermail/python-list/2006-August/371647.html
        https://pillow.readthedocs.io/en/stable/reference/Image.html
        :return:
        """
        image_str = pygame.image.tostring(self.game.screen, 'RGB')
        image = PIL.Image.frombytes(mode='RGB', size=(screen_height, screen_width), data=image_str)
        image_np_array = np.array(image)
        return image_np_array


class BounceHumanPlayRecord(gym.Env, PyGamePixelEnv):
    def __init__(self, program: Program, recording_dir=None, num_ball_to_win=1):

        self.program = program
        self.num_ball_to_win = num_ball_to_win

        self.recording_dir = recording_dir
        os.makedirs(recording_dir, exist_ok=True)

        self.bounce = Bounce(program)
        self.recorded_actions = []

        super().__init__(self.bounce)

    def run_loop(self, keys):
        self.bounce.paddle.stop_moving()  # always stop paddle running at first
        # record action sequence here
        if keys[pygame.K_LEFT]:
            action_index = self.bounce.action_cmds.index(pygame.K_LEFT)
        elif keys[pygame.K_RIGHT]:
            action_index = self.bounce.action_cmds.index(pygame.K_RIGHT)
        else:
            action_index = self.bounce.action_cmds.index(PAUSE)

        if self.bounce.when_left_arrow(keys):
            cmds = self.bounce.cc.remove_bounce(self.program[WHEN_LEFT_ARROW])
            list(map(lambda c: self.bounce.engine.execute(c), cmds))

        if self.bounce.when_right_arrow(keys):
            cmds = self.bounce.cc.remove_bounce(self.program[WHEN_RIGHT_ARROW])
            list(map(lambda c: self.bounce.engine.execute(c), cmds))

        # after a series of update...
        self.bounce.bg.draw(self.bounce.screen)

        self.bounce.sync_sprite()
        self.bounce.all_sprites.draw(self.bounce.screen)

        self.bounce.clock.tick(fps)
        self.bounce.space.step(1 / fps)

        self.bounce.score_board.draw(self.bounce.screen)

        pygame.display.flip()

        if self.recording_dir is not None:
            self.recorded_actions.append(action_index)  # self.get_image()

        return

    def run_with_max_skip(self, seed=None, max_len=None, max_skip=2):
        self.bounce.seed(seed)  # seeded! right before running
        self.bounce.when_run_execute()

        running = True
        iters = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # even at quitting, we are still executing this loop one last time...
                    running = False

            keys = pygame.key.get_pressed()
            # this lowers the refresh rate...
            for _ in range(max_skip):
                self.run_loop(keys)
                iters += 1
                if max_len:  # when this is not None
                    if iters >= max_len:
                        running = False  # and we let the saving happen

                # we also have a rule of cutting off
                if self.bounce.score_board.own == self.num_ball_to_win or self.bounce.score_board.opponent == self.num_ball_to_win:
                    running = False

        if not running:
            # we save here
            print("Number of frames: {}".format(iters))
            if self.recording_dir is not None:
                np.savez_compressed(
                    open(pjoin(self.recording_dir, "human_actions_{}_max_skip_{}_1_ball.npz".format(seed, max_skip)),
                         'wb'),
                    frames=np.array(self.recorded_actions, dtype=np.int))

                return pjoin(self.recording_dir, "human_actions_{}_max_skip_{}_1_ball.npz".format(seed, max_skip))

    def run(self, seed=None, max_len=None, debug=False):

        self.bounce.seed(seed)  # seeded! right before running
        self.bounce.when_run_execute()

        if debug:
            import pymunk
            draw_options = pymunk.pygame_util.DrawOptions(self.bounce.screen)

        running = True
        iters = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # even at quitting, we are still executing this loop one last time...
                    running = False

            self.bounce.paddle.stop_moving()  # always stop paddle running at first

            keys = pygame.key.get_pressed()

            # record action sequence here
            if keys[pygame.K_LEFT]:
                action_index = self.bounce.action_cmds.index(pygame.K_LEFT)
            elif keys[pygame.K_RIGHT]:
                action_index = self.bounce.action_cmds.index(pygame.K_RIGHT)
            else:
                action_index = self.bounce.action_cmds.index(PAUSE)

            if self.bounce.when_left_arrow(keys):
                cmds = self.bounce.cc.remove_bounce(self.program[WHEN_LEFT_ARROW])
                list(map(lambda c: self.bounce.engine.execute(c), cmds))

            if self.bounce.when_right_arrow(keys):
                cmds = self.bounce.cc.remove_bounce(self.program[WHEN_RIGHT_ARROW])
                list(map(lambda c: self.bounce.engine.execute(c), cmds))

            # after a series of update...
            self.bounce.bg.draw(self.bounce.screen)

            # draw PyMunk objects (don't need this when actually playing)
            if debug:
                self.bounce.space.debug_draw(draw_options)

            self.bounce.sync_sprite()
            self.bounce.all_sprites.draw(self.bounce.screen)

            self.bounce.clock.tick(fps)
            self.bounce.space.step(1 / fps)

            self.bounce.score_board.draw(self.bounce.screen)

            pygame.display.flip()

            if self.recording_dir is not None:
                self.recorded_actions.append(action_index)  # self.get_image()

            iters += 1
            if max_len:  # when this is not None
                if iters >= max_len:
                    running = False  # and we let the saving happen

            # we also have a rule of cutting off
            if self.bounce.score_board.own == self.num_ball_to_win or self.bounce.score_board.opponent == self.num_ball_to_win:
                running = False

        if not running:
            # we save here
            print("Number of frames: {}".format(iters))
            if self.recording_dir is not None:
                np.savez_compressed(open(pjoin(self.recording_dir, "human_actions_{}.npz".format(seed)), 'wb'),
                                    frames=np.array(self.recorded_actions, dtype=np.int))

                return pjoin(self.recording_dir, "human_actions_{}.npz".format(seed))


class BounceHumanPlayRecordVideo(gym.Env, PyGamePixelEnv):
    def __init__(self, program: Program, recording_dir=None):
        self.program = program

        self.bounce = Bounce(program)
        self.recording_dir = recording_dir
        os.makedirs(recording_dir, exist_ok=True)
        self.recorded_frames = []

        super().__init__(self.bounce)

    def run(self):

        self.bounce.when_run_execute()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    # we save here; this is fine...we save when we close the window
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

    def __init__(self, program: Program, reward_type, reward_shaping=False, num_ball_to_win=5,
                 finish_reward=100):

        assert reward_type in {ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO}
        self.reward_type = reward_type

        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.num_envs = 1
        self.viewer = None
        self.reward_shaping = reward_shaping
        self.num_ball_to_win = num_ball_to_win
        self.finish_reward = finish_reward

        self.prev_shaping = None

        self.program = program

        self.bounce = Bounce(program)
        self.action_space = define_action_space(self.bounce.action_cmds)
        self.observation_space = define_observation_space(screen_height, screen_width)

    def seed(self, seed=None):
        return self.bounce.seed(seed)

    def step(self, action):
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action >= len(self.bounce.action_cmds):
            raise Exception("Can't choose an action based on index {}".format(action))

        prev_score = self.bounce.score_board.own
        prev_oppo_score = self.bounce.score_board.opponent

        # map action into key
        action_key = self.bounce.action_cmds[action]

        self.bounce.act(action_key)

        score = self.bounce.score_board.own
        oppo_score = self.bounce.score_board.opponent

        score_diff = score - prev_score
        oppo_score_diff = oppo_score - prev_oppo_score

        done = False
        if score == self.num_ball_to_win or oppo_score == self.num_ball_to_win:
            done = True

        # considering "discount", reward should be higher
        # we can add reward shaping if necessary (through wrappers)
        if self.reward_type == SELF_MINUS_HALF_OPPO:
            reward = (score_diff - oppo_score_diff) * 20
            reward = max(-10, reward)
            # -10 vs. 20, we care more about goal than to catch the ball
            # assert reward in {20, -10, 0}  # highest reward is 100
        else:
            reward = score_diff * 20  # we only care about sending the ball in; each ball in is 1 point
            # full points: 100

        # Win everything: 200
        # Lose everything: -150
        if done and score == self.num_ball_to_win:
            reward += self.finish_reward  # 100  # + 120  (120 + 80 = 200)
        elif done and oppo_score == self.num_ball_to_win:
            reward -= self.finish_reward  # 100  # - 110 (-40 - 110 = -150)

        # reward shaping
        if self.reward_shaping:

            # only tracking one ball's distance
            ball_x, ball_y = self.bounce.ball_group.balls[0].body.position
            paddle_x, paddle_y = self.bounce.paddle.body.position

            shaping = - 100 * np.abs(ball_x - paddle_x) / 400

            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping

            self.prev_shaping = shaping

        # make reward a bit smaller...
        # reward /= 10

        return self.get_image(), reward, done, {"score": score, "oppo_score": oppo_score}

    def reset(self):
        # we take the shortcut -- re-create the instance
        # TODO: currently seeding needs to happen after reset
        # TODO: other ways seem to be worse...so do not consider it

        # seeding needs to happen after the reset
        self.bounce = Bounce(self.program)
        self.prev_shaping = None

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


class BounceObjectEnv(gym.Env):
    """
    We return paddle, ball_left, ball_top, gate_left, gate_right position
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, program: Program, reward_type, reward_shaping=False, num_ball_to_win=5,
                 finish_reward=100, no_neg_finish=False):

        assert reward_type in {ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO}
        self.reward_type = reward_type

        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        self.num_envs = 1
        self.viewer = None
        self.reward_shaping = reward_shaping
        self.num_ball_to_win = num_ball_to_win
        self.finish_reward = finish_reward
        self.no_neg_finish = no_neg_finish

        self.prev_shaping = None

        self.program = program

        self.bounce = Bounce(program)
        self.action_space = define_action_space(self.bounce.action_cmds)

        self.state_size = 3 + 2 * self.bounce.ball_group.LIMIT
        self.observation_space = define_object_observation_space(shape=(self.state_size,))

    def seed(self, seed=None):
        return self.bounce.seed(seed)

    def get_state(self):
        # multi-ball
        max_ball_num = self.bounce.ball_group.LIMIT
        # [paddle_left, goal_left, goal_right, ball_1_left, ball_1_top, ball_2_left ...]
        state = np.zeros(3 + 2 * max_ball_num)
        state[0] = self.bounce.paddle.body.position[0]
        state[1] = 100  # goal left's boundary
        state[2] = 300  # goal right's boundary
        cnt = 3
        for ball_id, ball in self.bounce.ball_group.balls.items():
            # out of the 10 balls limit, we break out of the loop
            if cnt >= self.state_size:
                break
            left, top = ball.body.position
            try:
                state[cnt] = left
                state[cnt+1] = top
            except:
                break
            cnt += 2
        return state

    def step(self, action):
        """
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if action >= len(self.bounce.action_cmds):
            raise Exception("Can't choose an action based on index {}".format(action))

        prev_score = self.bounce.score_board.own
        prev_oppo_score = self.bounce.score_board.opponent

        # map action into key
        action_key = self.bounce.action_cmds[action]

        self.bounce.act(action_key)

        score = self.bounce.score_board.own
        oppo_score = self.bounce.score_board.opponent

        score_diff = score - prev_score
        oppo_score_diff = oppo_score - prev_oppo_score

        done = False
        if score == self.num_ball_to_win or oppo_score == self.num_ball_to_win:
            done = True

        # considering "discount", reward should be higher
        # we can add reward shaping if necessary (through wrappers)
        if self.reward_type == SELF_MINUS_HALF_OPPO:
            reward = (score_diff - oppo_score_diff) * 20
            reward = max(-10, reward)
            # -10 vs. 20, we care more about goal than to catch the ball
            # assert reward in {20, -10, 0}  # highest reward is 100
        else:
            reward = score_diff * 20  # we only care about sending the ball in; each ball in is 1 point
            # full points: 100

        # Win everything: 200
        # Lose everything: -150
        if done and score == self.num_ball_to_win:
            reward += self.finish_reward  # 100  # + 120  (120 + 80 = 200)
        elif done and oppo_score == self.num_ball_to_win:
            if not self.no_neg_finish:
                reward -= self.finish_reward  # 100  # - 110 (-40 - 110 = -150)

        # reward shaping
        if self.reward_shaping:

            # only tracking one ball's distance
            ball_x, ball_y = self.bounce.ball_group.balls[0].body.position
            paddle_x, paddle_y = self.bounce.paddle.body.position

            shaping = - 100 * np.abs(ball_x - paddle_x) / 400

            if self.prev_shaping is not None:
                reward += shaping - self.prev_shaping

            self.prev_shaping = shaping

        # make reward a bit smaller...
        # reward /= 10

        return self.get_state(), reward, done, {"score": score, "oppo_score": oppo_score}

    def reset(self):
        # we take the shortcut -- re-create the instance
        # TODO: currently seeding needs to happen after reset
        # TODO: other ways seem to be worse...so do not consider it

        # seeding needs to happen after the reset
        self.bounce = Bounce(self.program)
        self.prev_shaping = None

        # the intitial state, except for paddle, should all be 0
        # this is not quite right, initial state should be given by the env
        # but in the case, we'll let it slide
        return np.zeros(self.state_size)

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


def interactive_run():
    program = Program()
    program.set_correct()
    # program.set_correct_with_theme()
    # program.load("../envs/bounce_programs/demo1.json")
    app = Bounce(program)
    app.run()


def record_human_play_to_video(program_name, video_name):
    assert ".mp4" in video_name
    assert ".json" in program_name

    # remove these lines for a more flexible function
    assert "/" not in program_name, "we prefix the directory, just enter the name of file"
    assert "/" not in video_name, "we prefix the directory, just enter the name of file"

    program = Program()

    program.load("../envs/bounce_programs/" + program_name)
    app = BounceHumanPlayRecordVideo(program, "./bounce_gameplay_recordings/")
    filename = app.run()

    convert_np_to_video(filename, "./bounce_gameplay_recordings/" + video_name)


def record_human_play_custom_program_to_video(program_json_str, video_name):
    assert ".mp4" in video_name

    # remove these lines for a more flexible function
    assert "/" not in video_name, "we prefix the directory, just enter the name of file"

    program = Program()

    program.loads(program_json_str)
    app = BounceHumanPlayRecordVideo(program, "./bounce_gameplay_recordings/")
    filename = app.run()

    convert_np_to_video(filename, "./bounce_gameplay_recordings/" + video_name)


def record_human_play_to_actions(program_name, seed, max_len=3000, max_skip=1):
    assert ".json" in program_name
    assert "/" not in program_name, "we prefix the directory, just enter the name of file"

    program = Program()

    program.load("../envs/bounce_programs/" + program_name)

    # the goal is to record some human play
    app = BounceHumanPlayRecord(program, "./bounce_humanplay_recordings/", num_ball_to_win=1)
    # we actually override initial positions after seeding
    # so printing won't tell us the true positions we sampled

    # print("assignment counter:", app.bounce.ball_group.assignment_counter)
    # print("Ball positions:", app.bounce.ball_group.ball_inits[:8])
    if max_skip != 1:
        app.run_with_max_skip(seed, max_len=max_len, max_skip=max_skip)
    else:
        app.run(seed, max_len=max_len)


def replay_human_play(program_name, human_play_npz, seed, max_len=3000):
    # this one will be a bit tricky
    # We will actually load in the Agent (Gym) environment
    # Luckily DQN doesn't require a vec env...so we are lucky here, loading in human will work fine

    assert '.npz' in human_play_npz

    program = Program()
    program.load("../envs/bounce_programs/" + program_name)
    env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, False, num_ball_to_win=1, finish_reward=0)

    from autograde.rl_envs.utils import SmartImageViewer

    human_actions = np.load("./bounce_humanplay_recordings/" + human_play_npz)['frames']  # this is a misnomer

    viewer = SmartImageViewer()

    obs = env.reset()
    env.seed(seed)  # NOTE: seed needs to happen after reset (unfortunate, I know...)

    # env.bounce.ball_group.get_new_ball_init()
    # print("assignment counter:", env.bounce.ball_group.assignment_counter)
    # print("Ball positions:", env.bounce.ball_group.ball_inits[:8])

    for i in range(max_len):
        # action = np.random.randint(env.action_space.n, size=1)

        obs, rewards, dones, info = env.step(human_actions[i])
        # env.render()
        # obs = obs.squeeze(0)
        viewer.imshow(obs)
        if rewards != 0:
            print(rewards)

        if dones:
            break

        env.bounce.clock.tick(fps)

    env.close()


def grouped(iterable, n):
    "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
    return zip(*[iter(iterable)] * n)


def verify_and_convert_human_play_to_max_skip(human_play_npz, seed, max_len=3000, max_skip=2):
    human_actions = np.load("./bounce_humanplay_recordings/" + human_play_npz)['frames']  # this is a misnomer
    print("first check if it's divisible by max_skip, or we have to pad the end: {}".format(
        human_actions.shape[0] % max_skip == 0))
    print("number of human actions:", human_actions.shape[0])

    converted_actions = []
    # now we check if all consecutive (max_skip) actions are the same
    global_counter = 0
    for tup_seg in grouped(human_actions.tolist(), max_skip):
        prev_a = None
        for i in range(max_skip):
            if prev_a is None:
                prev_a = tup_seg[i]
            else:
                if tup_seg[i] != prev_a:
                    print("pair index {}, actions={}, {}".format(global_counter, prev_a, tup_seg[i]))
                prev_a = tup_seg[i]

        converted_actions.append(tup_seg[0])
        global_counter += 1

    assert len(converted_actions) == human_actions.shape[0] / 2
    np.savez_compressed(open(
        pjoin("./bounce_humanplay_recordings/",
              "human_actions_{}_max_skip_{}_1_ball_converted.npz".format(seed, max_skip)),
        'wb'), frames=np.array(converted_actions, dtype=np.int))


def replay_human_play_with_sticky_actions(program_name, human_play_npz, seed, max_len=3000, max_skip=2):
    assert '.npz' in human_play_npz

    program = Program()
    program.load("../envs/bounce_programs/" + program_name)
    env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, True, num_ball_to_win=1, finish_reward=0)

    from autograde.rl_envs.utils import SmartImageViewer

    human_actions = np.load("./bounce_humanplay_recordings/" + human_play_npz)['frames']  # this is a misnomer

    viewer = SmartImageViewer()

    obs = env.reset()
    env.seed(seed)  # NOTE: seed needs to happen after reset (unfortunate, I know...)

    # env.bounce.ball_group.get_new_ball_init()
    # print("assignment counter:", env.bounce.ball_group.assignment_counter)
    # print("Ball positions:", env.bounce.ball_group.ball_inits[:8])

    reward = 0

    for i in range(max_len):
        # action = np.random.randint(env.action_space.n, size=1)
        action = human_actions[i]
        for _ in range(max_skip):
            obs, rewards, dones, info = env.step(action)
            # env.render()
            # obs = obs.squeeze(0)
            viewer.imshow(obs)
            reward += rewards
            # if rewards != 0:
            #     print(rewards)

            if dones:
                break

        if dones:
            break
        env.bounce.clock.tick(fps)

    print(reward)
    env.close()


def try_reward_shaping():
    import random
    from autograde.rl_envs.utils import SmartImageViewer

    max_len = 200

    program = Program()
    program.set_correct()
    env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, False, num_ball_to_win=1, finish_reward=0)

    obs = env.reset()
    viewer = SmartImageViewer()

    prev_shaping = None
    rewards = 0

    for i in range(max_len):
        # action = np.random.randint(env.action_space.n, size=1)
        action = random.choice([0, 1, 2])
        obs, reward, dones, info = env.step(action)
        # env.render()
        # obs = obs.squeeze(0)
        viewer.imshow(obs)

        ball_x, ball_y = env.bounce.ball_group.balls[0].body.position
        paddle_x, paddle_y = env.bounce.paddle.body.position

        # print(ball_x, ball_y, 'paddle', paddle_x, paddle_y)
        shaping = - 100 * np.abs(ball_x - paddle_x) / 400

        if prev_shaping is not None:
            reward += shaping - prev_shaping

        prev_shaping = shaping

        rewards += reward

        if dones:
            break
        env.bounce.clock.tick(fps)

    env.close()
    print(i, rewards)


def record_invariances():
    from string import Template

    random_program_str = Template("""
        {"when run": ["launch new ball", "set scene '${scene}'", "set ball '${ball}'", "set paddle '${paddle}'"],
          "when left arrow": ["move left"],
          "when right arrow": ["move right"],
          "when ball hits paddle": ["bounce ball"],
          "when ball hits wall": ["bounce ball"],
          "when ball in goal": ["score point", "launch new ball"],
          "when ball misses paddle": ["score opponent point",
                                      "launch new ball"]}
        """)

    # retro retro retro
    # retro retro normal
    # retro normal retro
    # normal retro retro
    # retro normal normal
    # normal retro normal
    # retro normal normal
    # normal normal normal

    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="retro", ball="retro", paddle="retro"),
    #                                           "retro.mp4")

    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="retro", ball="retro", paddle="hardcourt"),
    #                                           "retro_retro_hardcourt.mp4")
    #
    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="retro", ball="hardcourt", paddle="retro"),
    #                                           "retro_hardcourt_retro.mp4")

    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="hardcourt", ball="retro", paddle="retro"),
    #                                           "hardcourt_retro_retro.mp4")

    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="retro", ball="hardcourt", paddle="hardcourt"),
    #                                           "retro_normal_normal.mp4")

    record_human_play_custom_program_to_video(random_program_str.substitute(scene="hardcourt", ball="hardcourt", paddle="retro"),
                                              "hardcourt_hardcourt_retro.mp4")

    record_human_play_custom_program_to_video(random_program_str.substitute(scene="hardcourt", ball="retro", paddle="hardcourt"),
                                              "hardcourt_retro_hardcourt.mp4")

    # record_human_play_custom_program_to_video(random_program_str.substitute(scene="hardcourt", ball="hardcourt", paddle="hardcourt"),
    #                                           "normal.mp4")

def record_action_invariance():
    from string import Template

    random_program_str = Template("""
            {"when run": ["launch new ball"],
              "when left arrow": ["move left"],
              "when right arrow": ["move right"],
              "when ball hits paddle": ["bounce ball"],
              "when ball hits wall": ["bounce ball", "set scene '${scene}'", "set ball '${ball}'", "set paddle '${paddle}'"],
              "when ball in goal": ["score point", "launch new ball"],
              "when ball misses paddle": ["score opponent point",
                                          "launch new ball"]}
            """)

    record_human_play_custom_program_to_video(random_program_str.substitute(scene="random", ball="random", paddle="random"),
                                              "action_invariance.mp4")

if __name__ == '__main__':

    pass
    # use this to play the game as a human
    # interactive_run()

    # record human play videos, and then convert them!
    # record_human_play_to_video("demo2.json", "video2.mp4")

    # reocrd human actions when playing
    # record_human_play_to_actions("correct_sample.json", seed=2222)  # do not override this...you got 5:0
    # record_human_play_to_actions("correct_sample.json", seed=2222, max_skip=2)
    # record_human_play_to_actions("correct_sample.json", seed=4141)

    # record_human_play_to_actions("correct_sample.json", seed=1414, max_skip=2, max_len=1000)
    # record_human_play_to_actions("correct_sample.json", seed=2121, max_skip=2, max_len=1000)
    # record_human_play_to_actions("correct_sample.json", seed=5555, max_skip=2, max_len=1000)

    # replay_human_play("correct_sample.json", "human_actions_2222.npz", seed=2222)

    # verify_and_convert_human_play_to_max_skip("human_actions_2222_max_skip_2.npz", seed=2222, max_skip=2)

    # verify_and_convert_human_play_to_max_skip("human_actions_1414_max_skip_2_1_ball.npz", seed=1414, max_skip=2)
    # verify_and_convert_human_play_to_max_skip("human_actions_2121_max_skip_2_1_ball.npz", seed=2121, max_skip=2)
    # verify_and_convert_human_play_to_max_skip("human_actions_5555_max_skip_2_1_ball.npz", seed=5555, max_skip=2)

    # so it should be 1500 time steps, not 3000...but check Monitor counts real frames or agent actions

    # replay_human_play_with_sticky_actions("correct_sample.json", "human_actions_2222_max_skip_2_converted.npz",
    #                                       seed=2222, max_skip=2)

    # try_reward_shaping()

    # record_invariances()
    # record_action_invariance()
