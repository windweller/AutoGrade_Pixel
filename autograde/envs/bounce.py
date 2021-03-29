"""
We are making a Bounce game where it's modularized
useing PyMunk for physics engine

Situations:
#1: when run
#2: when left arrow
#3: when right arrow
#4: when ball hits paddle
#5: when ball hits wall
#6: when ball in goal
#7: when ball misses paddle

Conditions:
(Add as collision type)
Ball <-> Paddle
Ball <-> Wall
Ball <-> top line (in goal)
Ball <-> bottom line (miss paddle)

Actions:
#1: move left
#2: move right
#3: bounce ball
#4: score point
#5: score opponent point
#6: launch new ball
#7: set "normal" paddle speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']
#8: set "normal" ball speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']

#1: set "hardcourt" scene: ['hardcourt', 'retro', 'random']
#2: set "hardcourt" ball: ['hardcourt', 'retro', 'random']
#3: set "hardcourt" paddle: ['hardcourt', 'retro', 'random']

move-left and move-right can be triggered by 4 ball conditions and when-run
but we can just give it a visual update without adding velocity?

"""
import os
import pathlib

import pygame
import pymunk

from pygame.locals import *
import pymunk.pygame_util

import json
from abc import ABC, abstractmethod

try:
    from . import utils_seeding as seeding
except:
    import utils_seeding as seeding

# ==== Settings ====

HARDCOURT = 'hardcourt'
RETRO = 'retro'
RANDOM = 'random'

# we change speed to velocity instead
# and do one time update
speed_text_available = ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']
speed_choices = ['very slow', 'slow', 'normal', 'fast', 'very fast']  #
speed_dict = {
    'very slow': 100, 'slow': 200, 'normal': 300, 'fast': 400, 'very fast': 500
}

theme_choices = [HARDCOURT, RETRO]
theme_dict = {
    'retro': RETRO,
    'hardcourt': HARDCOURT,
    'random': RANDOM
}

collision_types = {
    "ball": 1,  # many balls
    "goal": 2,
    "bottom": 3,
    "paddle": 4,
    "wall": 5  # many walls
}

screen_width = 400
screen_height = 400
distance_from_boundary = 17
fps = 50
BALL_RADIUS = 14

wall_thickness = 15

game_name = "Bounce"
PAUSE = 'Pause'

# 1: move left
# 2: move right
# 3: bounce ball
# 4: score point
# 5: score opponent point
# 6: launch new ball
# 7: set "normal" paddle speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']
# 8: set "normal" ball speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']

# 1: set "hardcourt" scene: ['hardcourt', 'retro', 'random']
# 2: set "hardcourt" ball: ['hardcourt', 'retro', 'random']
# 3: set "hardcourt" paddle: ['hardcourt', 'retro', 'random']

MOVE_LEFT = "move left"
MOVE_RIGHT = "move right"
BOUNCE_BALL = "bounce ball"
SCORE_POINT = "score point"
SCORE_OPPO_POINT = "score opponent point"
LAUNCH_NEW_BALL = "launch new ball"

WHEN_RUN = 'when run'
WHEN_LEFT_ARROW = 'when left arrow'
WHEN_RIGHT_ARROW = 'when right arrow'

BALL_HIT_PADDLE = "when ball hits paddle"
BALL_HIT_WALL = "when ball hits wall"
BALL_IN_GOAL = "when ball in goal"
BALL_MISS_PADDLE = "when ball misses paddle"


def to_pygame(pos):
    return pos[0], screen_height - pos[1]


# =====================

class Config(object):
    def update(self):
        """
        This updates the self.config_dict, and set up attributes for the class
        :return:
        """
        for k, v in self.config_dict.items():
            self.__setattr__(k.replace(' ', "_"), v)

    def __getitem__(self, y):
        if "_" in y:
            y = y.replace("_", " ")
        return self.config_dict[y]

    def __setitem__(self, key, value):
        """
        :param key:
        :param value: be careful with value, it overrides things
        :return:
        """
        if "_" in key:
            key = key.replace("_", " ")
        self.config_dict[key] = value

    def save(self, file_name):
        json.dump(self.config_dict, open(file_name, "w"))

    def load(self, file_name):
        # this method overrides
        self.config_dict = json.load(open(file_name))
        self.update()

    def loads(self, json_obj):
        if type(json_obj) == str:
            result = None
            try:
                result = json.loads(json_obj)
            except:
                pass

            try:
                # we assume this is
                result = json.load(open(json_obj))
            except:
                pass

            assert result is not None, "We are not able to parse the obj you sent in"
            json_obj = result

        assert type(json_obj) == dict
        self.config_dict = json_obj
        self.update()


class Program(Config):
    def __init__(self):
        self.config_dict = {
            "when run": [],
            "when left arrow": [],
            "when right arrow": [],
            "when ball hits paddle": [],
            "when ball hits wall": [],
            "when ball in goal": [],
            "when ball misses paddle": []
        }
        self.update()

    def set_correct(self):
        # we generate a correct program
        self.config_dict['when run'].append(LAUNCH_NEW_BALL)
        self.config_dict['when left arrow'].append(MOVE_LEFT)
        self.config_dict['when right arrow'].append(MOVE_RIGHT)
        self.config_dict['when ball hits paddle'].append(BOUNCE_BALL)
        self.config_dict['when ball hits wall'].append(BOUNCE_BALL)
        self.config_dict['when ball in goal'].append(SCORE_POINT)
        self.config_dict['when ball in goal'].append(LAUNCH_NEW_BALL)
        self.config_dict['when ball misses paddle'].append(SCORE_OPPO_POINT)
        self.config_dict['when ball misses paddle'].append(LAUNCH_NEW_BALL)

    def set_correct_retro_theme(self):
        self.config_dict['when run'].append(LAUNCH_NEW_BALL)
        self.config_dict['when run'].append("set 'retro' scene")
        self.config_dict['when run'].append("set 'retro' ball")
        self.config_dict['when run'].append("set 'retro' paddle")
        self.config_dict['when left arrow'].append(MOVE_LEFT)
        self.config_dict['when right arrow'].append(MOVE_RIGHT)
        self.config_dict['when ball hits paddle'].append(BOUNCE_BALL)
        self.config_dict['when ball hits wall'].append(BOUNCE_BALL)
        self.config_dict['when ball in goal'].append(SCORE_POINT)
        self.config_dict['when ball in goal'].append(LAUNCH_NEW_BALL)
        self.config_dict['when ball misses paddle'].append(SCORE_OPPO_POINT)
        self.config_dict['when ball misses paddle'].append(LAUNCH_NEW_BALL)


# ===== Pygame/PyMunk =======

class Theme(object):

    def __init__(self):
        curr_dir = pathlib.Path(__file__).parent.absolute()
        hardcourt_addr = {
            'background': os.path.join(curr_dir, 'bounce_assets/background.png'),
            'paddle': os.path.join(curr_dir, './bounce_assets/paddle.png'),
            'wall': os.path.join(curr_dir, './bounce_assets/wall.png'),
            'ball': os.path.join(curr_dir, './bounce_assets/ball.png'),
        }
        retro_addr = {
            'background': os.path.join(curr_dir, './bounce_assets/retro_background.png'),
            'paddle': os.path.join(curr_dir, './bounce_assets/retro_paddle.png'),
            'wall': os.path.join(curr_dir, './bounce_assets/retro_wall.png'),
            'ball': os.path.join(curr_dir, './bounce_assets/retro_ball.png')
        }

        self.asset_addr = {
            HARDCOURT: hardcourt_addr,
            RETRO: retro_addr
        }

    def get_img_path(self, theme, name):
        assert theme in {HARDCOURT, RETRO}
        assert name in self.asset_addr[HARDCOURT].keys()

        return self.asset_addr[theme][name]


theme_control = Theme()


class AbstractGameObject(ABC):
    # @abstractmethod
    # def set_theme(self, theme_str):
    #     raise NotImplementedError

    # @abstractmethod
    # def destroy(self):
    #     raise NotImplementedError

    @abstractmethod
    def sync_sprite(self):
        raise NotImplementedError

    @abstractmethod
    def create(self):
        raise NotImplementedError


class Background(object):
    def __init__(self):
        self.curr_theme = HARDCOURT
        self.create()

    def set_theme(self, theme_str):
        if theme_str != self.curr_theme:
            # need to load paddle's last position, recreate it
            self.curr_theme = theme_str
            self.create()

    def draw(self, screen):
        # this is the blit method
        screen.blit(self.sprite, (0, 0))

    def create(self):
        img_path = theme_control.get_img_path(self.curr_theme, 'background')
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (screen_width, screen_height))
        self.sprite = img.convert()


def pymunk_safe_remove(space, body, shape):
    if shape is not None:
        if shape in space.shapes:
            space.remove(shape)
    if body is not None:
        if body in space.bodies:
            space.remove(body)


class Ball(AbstractGameObject):
    def __init__(self, id, space, all_sprites, np_random, theme_str=None,
                 position=None, direction=None):
        """
        This will add a ball to space, all_sprites
        and save a reference to it

        Since we will remove balls by arbitor, we need to look up the sprite
        This means, we need to link ball and sprite (through look up)

        :param space: PyMunk space
        :param all_sprites: PyGame sprites Group class
        """
        self.exists = False  # call this to learn if this is destroyed
        self.id = id

        self.space = space
        self.all_sprites = all_sprites

        self.curr_theme = HARDCOURT if theme_str is None else theme_str

        self.sprite = None

        self.radius = BALL_RADIUS # 14  # 28x28
        self.speed = speed_dict['normal']

        self.np_random = np_random

        self.create(position, direction)

    def create(self, position=None, direction=None):
        # seperate for HARDCOURT and RETRO
        # this is only used to launch a new ball; change theme only changes sprite
        ball_body = pymunk.Body(1, pymunk.inf)

        # ball_body.position = (
        #     random.randint(self.radius + wall_thickness, screen_width - wall_thickness - self.radius), 300)
        if position is not None:
            ball_body.position = position
        else:
            ball_body.position = (
                self.np_random.randint(self.radius + wall_thickness, screen_width - wall_thickness - self.radius), 300)

        # Poly is better to accommodate the angle
        ball_shape = pymunk.Poly.create_box(ball_body, (self.radius * 2, self.radius * 2))
        # ball_shape = pymunk.Circle(ball_body, self.radius)

        # direction = random.choice(
        #     [(random.randint(1, self.speed / 2), -self.speed), (random.randint(-self.speed / 2, -1), -self.speed)])
        if direction is None:
            choice_idx = self.np_random.choice([0, 1])
            direction = [(self.np_random.randint(1, int(self.speed / 2)), -self.speed),
                 (self.np_random.randint(-int(self.speed / 2), -1), -self.speed)][choice_idx]

        ball_body.apply_impulse_at_local_point(pymunk.Vec2d(direction))

        ball_shape.elasticity = 1.0
        ball_shape.collision_type = collision_types["ball"]

        self.body = ball_body
        self.shape = ball_shape

        self.shape.id = self.id

        self.space.add(self.body, self.shape)

        self.sprite = self.create_sprite(self.body.position)
        self.all_sprites.add(self.sprite)

        self.exists = True

    def create_sprite(self, pos):
        if self.curr_theme == HARDCOURT:
            scale_size = (30, 30)  # 30x30 (because we have "shadow")
        else:
            scale_size = (27, 27)
        img_path = theme_control.get_img_path(self.curr_theme, 'ball')
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, scale_size)

        sprite = pygame.sprite.Sprite()
        sprite.image = img.convert_alpha()
        sprite.rect = img.get_rect()
        sprite.rect.center = to_pygame(pos)

        return sprite

    def destroy(self):
        # IMPORTANT: cannot call this function inside a callback
        # Chipmunk will throw an error
        # but this function is useful for set_theme(), hence kept here
        self.all_sprites.remove(self.sprite)
        self.space.remove(self.shape.body, self.shape)
        self.exists = False

    def destroy_sprite(self):
        self.all_sprites.remove(self.sprite)
        self.exists = False

    def set_theme(self, theme_str):
        if theme_str != self.curr_theme:
            # we are not destroying the physical body, but just the sprite
            self.curr_theme = theme_str
            # not destroying anything
            # but replace the sprite
            self.all_sprites.remove(self.sprite)
            # load in another
            pos = self.body.position
            self.sprite = self.create_sprite(pos)
            self.all_sprites.add(self.sprite)

    def sync_sprite(self):
        self.sprite.rect.center = to_pygame(self.body.position)

    def normalize_velocity(self):
        # this will be called by CollisionHandler
        self.body.velocity = self.body.velocity * (self.speed / self.body.velocity.length)

    def set_speed(self, new_speed):
        self.speed = speed_dict[new_speed]
        self.normalize_velocity()


class BallGroup(object):
    def __init__(self, space, all_sprites, np_random):
        self.balls = {}  # {ball_name: ball_obj}  (ball name is passed into PyMunk object)

        self.exist_balls = {}
        # this is used to control sprite
        self.space = space
        self.all_sprites = all_sprites

        self.curr_theme = HARDCOURT
        self.curr_ball_num = 0
        self.LIMIT = 10

        self.np_random = np_random
        self.speed = speed_dict['normal']

        # in order to solve randomness problem
        # we actually sample a fixed set of ball directions and positions
        # and then we loop on them
        # we sample 1000
        self.num_inits_stored = 100  # 10 balls appearing 10 times
        self.assignment_counter = 0
        self.ball_inits = self.sample_ball_inits()
        self.curr_seed = np_random.curr_seed

    def sample_ball_inits(self):
        # can't use map -- it destroys our random stuff
        res = []
        for _ in range(self.num_inits_stored):
            res.append(self.sample_ball_init())
        return res

    def get_new_ball_init(self):
        # change of seed will result in a new sampling
        # and reset of counter
        if self.np_random.curr_seed != self.curr_seed:
            self.ball_inits = self.sample_ball_inits()
            self.assignment_counter = 0

        # no-boundary looping is here
        if self.assignment_counter >= self.num_inits_stored:
            # reset to 0 when reach 100
            self.assignment_counter = 0

        new_init = self.ball_inits[self.assignment_counter]
        self.assignment_counter += 1
        return new_init

    def sample_ball_init(self):
        position = (
            self.np_random.randint(BALL_RADIUS + wall_thickness, screen_width - wall_thickness - BALL_RADIUS), 300)
        choice_idx = self.np_random.choice([0, 1])
        direction = [(self.np_random.randint(1, int(self.speed / 2)), -self.speed),
                     (self.np_random.randint(-int(self.speed / 2), -1), -self.speed)][choice_idx]

        return position, direction

    def exists(self, id):
        return id in self.exist_balls

    def get_ball_ids(self):
        return self.balls.keys()

    def add(self, id=None):
        if id is None:
            id = len(self.balls)
        if self.curr_ball_num <= self.LIMIT:
            # get new init
            position, direction = self.get_new_ball_init()
            self.balls[id] = Ball(id, self.space, self.all_sprites, self.np_random, self.curr_theme,
                                  position, direction)
            self.curr_ball_num += 1
            self.exist_balls[id] = self.balls[id]
        else:
            pass

    def get(self, id):
        return self.balls[id]

    def sync_sprite(self):
        list(map(lambda tup: tup[1].sync_sprite(), self.exist_balls.items()))

    def set_theme(self, theme_str):
        if theme_str != self.curr_theme:
            self.curr_theme = theme_str
            list(map(lambda tup: tup[1].set_theme(theme_str), self.exist_balls.items()))

    def normalize_velocity(self):
        list(map(lambda tup: tup[1].normalize_velocity(), self.exist_balls.items()))

    def remove(self, id):
        # we only destroy the ball, but do not delete it from the list!
        # this is for the NN perspective
        # self.balls[id].destroy()

        # the PyMunk object is deleted in Controller already

        # there are scenarios where we are destroying balls that should already be gone!
        # TODO: hmmm, wrong, so after 10 balls, no new balls? I guess for a 5:5 game, this works out
        # TODO: but logic is wrong here
        if id in self.exist_balls:
            self.balls[id].destroy_sprite()
            del self.exist_balls[id]
            self.curr_ball_num -= 1

    def set_speed(self, new_speed):
        # we don't need to fix random seed here...I hope
        if new_speed == 'random':
            new_speed = self.np_random.choice(list(speed_dict.keys()))

        self.speed = speed_dict[new_speed]
        list(map(lambda tup: tup[1].set_speed(new_speed), self.exist_balls.items()))


class Wall(object):
    def __init__(self, space, pos, width, height):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = pos

        self.shape = pymunk.Poly.create_box(self.body, (width, height))

        self.shape.elasticity = 1
        self.shape.collision_type = collision_types['wall']

        space.add(self.shape, self.body)


class WallGroup(AbstractGameObject):
    def __init__(self, space, all_sprites):
        wall_thickness = 15
        wall_left = Wall(space, [0, screen_height / 2], wall_thickness, screen_height)
        wall_right = Wall(space, [screen_width, screen_height / 2], wall_thickness, screen_height)
        wall_top_left = Wall(space, [50, screen_height], 100, wall_thickness)
        wall_top_right = Wall(space, [screen_width - 50, screen_height], 100, wall_thickness)

        self.sprite = None
        self.curr_theme = HARDCOURT

        self.all_sprites = all_sprites

        self.create()

    def create(self):
        self.sprite = self.create_sprite([screen_width / 2, screen_height / 2])
        self.all_sprites.add(self.sprite)

    def create_sprite(self, pos):
        img_path = theme_control.get_img_path(self.curr_theme, 'wall')
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (screen_width, screen_height))

        sprite = pygame.sprite.Sprite()
        sprite.image = img.convert_alpha()
        sprite.rect = img.get_rect()
        sprite.rect.center = to_pygame(pos)

        return sprite

    def set_theme(self, theme_str):
        if theme_str != self.curr_theme:
            # need to load paddle's last position, destroy, then recreate it
            self.curr_theme = theme_str
            self.destroy_sprite()
            self.create()

    def sync_sprite(self):
        pass

    def destroy_sprite(self):
        self.all_sprites.remove(self.sprite)

    def destroy(self):
        pass


class Paddle(AbstractGameObject):
    """
    PyMunk a circle, but sprite is only semi-circle
    """

    def __init__(self, space, all_sprites):
        self.curr_theme = HARDCOURT
        self.init_pos = [screen_width / 2, distance_from_boundary]  # screen_height -
        self.space = space
        self.all_sprites = all_sprites

        self.hardcourt_pymunk_y = self.init_pos[1] - 16
        self.retro_pymunk_y = self.init_pos[1]

        self.body, self.shape, self.sprite = None, None, None

        self.speed = speed_dict['normal']

        self.create()

    def set_theme(self, theme_str):
        if theme_str != self.curr_theme:
            # need to load paddle's last position, destroy, then recreate it
            self.curr_theme = theme_str
            pos = self.body.position

            self.destroy_sprite()
            self.sprite = self.create_sprite(pos)
            self.all_sprites.add(self.sprite)

            self.curr_theme = theme_str
            # not destroying anything
            # but replace the sprite
            self.all_sprites.remove(self.sprite)

            #self.destroy()
            #self.create(curr_pos)

    def set_speed(self, new_speed):
        self.speed = speed_dict[new_speed]

    def move_left(self):
        self.start_moving(-self.speed)

    def move_right(self):
        self.start_moving(self.speed)

    def start_moving(self, vel):
        if self.on_edge() == 'left' and vel < 0:
            self.reset_on_edge()
        elif self.on_edge() == 'right' and vel > 0:
            self.reset_on_edge()
        else:
            self.body.velocity = vel, 0

    def stop_moving(self):
        self.body.velocity = 0, 0

    def get_pos(self):
        return self.body.position

    def create_sprite(self, pos):

        img_path = theme_control.get_img_path(self.curr_theme, 'paddle')
        img = pygame.image.load(img_path)

        if self.curr_theme == HARDCOURT:
            img = pygame.transform.scale(img, (45, 20))
        else:
            img = pygame.transform.scale(img, (43, 16))

        sprite = pygame.sprite.Sprite()
        sprite.image = img.convert_alpha()
        sprite.rect = img.get_rect()
        sprite.rect.center = to_pygame(pos)

        return sprite

    def create(self, pos=None):

        pos = self.init_pos if pos is None else pos
        self.sprite = self.create_sprite(pos)

        paddle_body = pymunk.Body(500, pymunk.inf, pymunk.Body.KINEMATIC)

        if self.curr_theme == HARDCOURT:
            radius = 25
            # then pos is actually different for pymunk, because center is no longer (400, 50), it's (400, 5)
            paddle_body.position = [pos[0], self.hardcourt_pymunk_y]
            paddle_shape = pymunk.Circle(paddle_body, radius)
        else:
            paddle_body.position = [pos[0], self.retro_pymunk_y]
            paddle_shape = pymunk.Poly.create_box(paddle_body, (43, 16))  # 44x16, 22x8

        paddle_shape.elasticity = 1
        paddle_shape.collision_type = collision_types['paddle']

        self.shape = paddle_shape
        self.body = paddle_body

        self.space.add(self.body, self.shape)
        self.all_sprites.add(self.sprite)

    def on_edge(self):
        if self.body.position[0] <= 25:
            return 'left'
        elif self.body.position[0] >= screen_width - 25:
            return 'right'
        else:
            return False

    def reset_on_edge(self):
        # ignore move command, just reset
        if self.body.position[0] <= 25:
            self.body.velocity = 0, 0
            y = self.hardcourt_pymunk_y if self.curr_theme == HARDCOURT else self.retro_pymunk_y
            self.body.position = [25, y]
        elif self.body.position[0] >= screen_width - 25:
            self.body.velocity = 0, 0
            y = self.hardcourt_pymunk_y if self.curr_theme == HARDCOURT else self.retro_pymunk_y
            self.body.position = [screen_width - 25, y]

    def destroy(self):
        # later on, we will use
        self.space.remove(self.shape.body, self.shape)
        self.all_sprites.remove(self.sprite)
        self.body, self.shape, self.sprite = None, None, None

    def destroy_sprite(self):
        self.all_sprites.remove(self.sprite)

    def sync_sprite(self):
        # remember that the paddle height and pymunk ball height aren't the same
        self.sprite.rect.center = [self.body.position[0], screen_height - self.init_pos[1]]


class Goal(object):
    def __init__(self, space, all_sprites):
        self.all_sprites = all_sprites
        self.space = space

        # top = pymunk.Segment(space.static_body, (100, screen_height + 2), (300, screen_height + 2), 2)
        top = pymunk.Segment(space.static_body, (-500, screen_height + 7), (500, screen_height + 7), 2)
        # top = pymunk.Segment(space.static_body, (0, screen_height + 7), (500, screen_height + 7), 2)
        top.elasticity = 1.0
        top.collision_type = collision_types['goal']

        self.shape = top
        self.space.add(top)


class Bottom(object):
    def __init__(self, space, all_sprites):
        self.all_sprites = all_sprites
        self.space = space

        # currently this is outside of screen
        bottom = pymunk.Segment(space.static_body, (-500, -15), (500, -15), 1)
        bottom.sensor = True  # miss paddle is a true sensor (you can't bounce on it)
        bottom.collision_type = collision_types["bottom"]

        self.shape = bottom
        self.space.add(bottom)


class ScoreBoard(object):
    def __init__(self):
        self.own = 0
        self.opponent = 0

        self.font = pygame.font.SysFont("comicsansms", 25)
        self.white = pygame.Color(255, 255, 255)

    def score_point(self):
        self.own += 1

    def score_oppo_point(self):
        self.opponent += 1

    def draw(self, screen):
        text = "Score " + str(self.own) + " : " + str(self.opponent)
        rendered_text = self.font.render(text, True, self.white)
        x_pos = (screen_width - rendered_text.get_width()) // 2
        screen.blit(rendered_text, (x_pos, 50))


class ShadowEngine(object):
    def __init__(self, ball_group: BallGroup,
                 paddle: Paddle, goal: Goal,
                 bottom: Bottom, background: Background,
                 wall_group: WallGroup,
                 score_board: ScoreBoard,
                 np_random):
        self.ball_group = ball_group
        self.paddle = paddle
        self.bottom = bottom
        self.goal = goal
        self.background = background
        self.score_board = score_board
        self.wall_group = wall_group

        self.np_random = np_random

        self.executable_cmds = [MOVE_LEFT, MOVE_RIGHT, SCORE_OPPO_POINT, SCORE_POINT, LAUNCH_NEW_BALL]

    def execute(self, cmd):
        # we handle bounce in collision controller
        assert cmd != "bounce ball"
        if cmd in self.executable_cmds:
            if " " in cmd:
                cmd = cmd.replace(" ", "_")
            eval("self." + cmd + "()")
        elif "paddle speed" in cmd:
            self.set_paddle_speed(cmd)
        elif "ball speed" in cmd:
            self.set_ball_speed(cmd)
        elif 'scene' in cmd:
            self.set_scene_theme(cmd)
        elif 'ball' in cmd:
            self.set_ball_theme(cmd)
        elif 'paddle' in cmd:
            self.set_paddle_theme(cmd)

    def move_left(self):
        # move left doesn't mean actually move left (in original game)
        # just means set left speed for 1 frame
        self.paddle.move_left()

    def move_right(self):
        self.paddle.move_right()

    def score_point(self):
        self.score_board.score_point()

    def score_opponent_point(self):
        self.score_board.score_oppo_point()

    def launch_new_ball(self):
        self.ball_group.add()

    def set_scene_theme(self, cmd):
        theme = self.extract_theme(cmd)
        self.background.set_theme(theme)
        self.wall_group.set_theme(theme)

    def set_ball_theme(self, cmd):
        theme = self.extract_theme(cmd)
        self.ball_group.set_theme(theme)

    def set_paddle_theme(self, cmd):
        theme = self.extract_theme(cmd)
        self.paddle.set_theme(theme)

    def set_paddle_speed(self, cmd):
        speed = self.extract_speed(cmd)
        self.paddle.set_speed(speed)

    def set_ball_speed(self, cmd):
        speed = self.extract_speed(cmd)
        self.ball_group.set_speed(speed)

    def extract_speed(self, cmd):
        speed_text = cmd.split("'")[1]
        assert speed_text in speed_text_available, "the speed test used is {}".format(speed_text)

        if speed_text == 'random':
            speed_text = self.np_random.choice(speed_choices)  # choose a non-random option

        assert speed_text in speed_choices
        return speed_text

    def extract_theme(self, cmd):
        theme_text = cmd.split("'")[1]
        if theme_dict[theme_text] == 'random':
            theme_text = self.np_random.choice(theme_choices)
        return theme_text


class CollisionController(object):
    def __init__(self, space, program, ball_group, engine: ShadowEngine, np_random):

        self.np_random = np_random
        self.ball_group = ball_group
        self.program = program
        self.engine = engine

        self.ball_to_wall_collision_handler = space.add_collision_handler(collision_types['ball'],
                                                                          collision_types['wall'])
        self.ball_to_paddle_collision_handler = space.add_collision_handler(collision_types['ball'],
                                                                            collision_types['paddle'])
        self.ball_to_bottom_collision_handler = space.add_collision_handler(collision_types['ball'],
                                                                            collision_types['bottom'])
        self.ball_to_goal_collision_handler = space.add_collision_handler(collision_types['ball'],
                                                                          collision_types['goal'])

        # self.ball_to_ball_ch = space.add_collision_handler(collision_types['ball'], collision_types['ball'])
        # self.ball_to_ball_ch.begin = self.get_no_bounce()

        # Normalize velocity
        self.ball_to_paddle_collision_handler.post_solve = self.normalize_velocity()

        # after missing paddle, remove ball (only after overlapping stops)
        self.ball_to_bottom_collision_handler.separate = self.ball_through_and_remove()

        self.handler_dict = {
            BALL_HIT_WALL: self.ball_to_wall_collision_handler,
            BALL_HIT_PADDLE: self.ball_to_paddle_collision_handler,
            BALL_IN_GOAL: self.ball_to_goal_collision_handler,
            BALL_MISS_PADDLE: self.ball_to_bottom_collision_handler
        }

        self.set_no_bounce(BALL_IN_GOAL)
        self.set_no_bounce(BALL_HIT_PADDLE, remove=False)
        self.set_no_bounce(BALL_HIT_WALL)

    def set_no_bounce(self, condition, remove=True):
        self.handler_dict[condition].begin = self.get_no_bounce()
        if remove:
            self.handler_dict[condition].separate = self.ball_through_and_remove()

    def get_no_bounce(self):
        def no_bounce(arbiter, space, data):
            return False

        return no_bounce

    def set_bounce(self):
        def bounce(arbiter, space, data):
            return True

        return bounce

    def set_bounce_to_handler(self, condition):
        assert condition != BALL_MISS_PADDLE, "Missing paddle will not incur bounce"
        self.handler_dict[condition].begin = self.set_bounce()
        self.handler_dict[condition].post_solve = self.normalize_velocity()
        self.handler_dict[condition].separate = self.get_empty_func()

    def get_empty_func(self, bounce=True):
        def empty_fn(arbiter, space, data):
            return bounce

        return empty_fn

    def ball_through_and_remove(self):
        def rem_ball(arbiter, space, data):
            ball_shape = arbiter.shapes[0]
            self.ball_group.remove(ball_shape.id)
            if self.ball_group.exists(ball_shape.id):
                space.remove(ball_shape, ball_shape.body)  # ball_shape.body

            return False

        return rem_ball

    def normalize_velocity(self):
        # in a callback, we can still access environment variable in scope
        def norm_vel(arbiter, space, data):
            self.ball_group.normalize_velocity()
            return True

        return norm_vel

    def get_program_to_callback(self, condition, commands, bounce_or_not):
        # handle "bounce" separately
        # iterate through cmds...
        def callback(arbiter, space, data):
            for cmd in commands:
                self.engine.execute(cmd)
            return bounce_or_not

        return callback

    def remove_bounce(self, cmds):
        return [c for c in cmds if c != BOUNCE_BALL]

    def isolate_set_cmds(self, cmds):
        return [c for c in cmds if 'set' not in c], [c for c in cmds if 'set' in c]

    # 1: move left
    # 2: move right
    # 3: bounce ball
    # 4: score point
    # 5: score opponent point
    # 6: launch new ball
    # 7: set "normal" paddle speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']
    # 8: set "normal" ball speed: ['random', 'very slow', 'slow', 'normal', 'fast', 'very fast']

    # 1: set "hardcourt" scene: ['hardcourt', 'retro', 'random']
    # 2: set "hardcourt" ball: ['hardcourt', 'retro', 'random']
    # 3: set "hardcourt" paddle: ['hardcourt', 'retro', 'random']

    def wrap(self, func, prev_callback):
        def new_call_back(arbiter, space, data):
            prev_callback(arbiter, space, data)
            func(arbiter, space, data)

        return new_call_back

    # let's compress this into one function after we examine there's no problem
    def compile(self):
        # this compiles the whole program
        # Let's leave post_solve to normalize_vec
        cmds = self.program[BALL_HIT_WALL]
        cmds, set_cmds = self.isolate_set_cmds(cmds)
        if BOUNCE_BALL in cmds:
            # if bounces, then we don't need fall_through_and_remove
            # so everything else is fine
            cmds = self.remove_bounce(cmds)
            callback_func = self.get_program_to_callback(BALL_HIT_WALL, cmds, bounce_or_not=True)
            self.handler_dict[BALL_HIT_WALL].begin = callback_func
            self.handler_dict[BALL_HIT_WALL].post_solve = self.normalize_velocity()

            set_callback_func = self.get_program_to_callback(BALL_HIT_WALL, set_cmds, True)
            self.handler_dict[BALL_HIT_WALL].separate = set_callback_func  # self.get_empty_func()
        else:
            callback_func = self.get_program_to_callback(BALL_HIT_WALL, cmds, bounce_or_not=False)
            self.handler_dict[BALL_HIT_WALL].begin = callback_func
            self.handler_dict[BALL_HIT_WALL].separate = self.ball_through_and_remove()

        # ======== Miss paddle ===========
        cmds = self.program[BALL_MISS_PADDLE]
        cmds = self.remove_bounce(cmds)

        # cmds, set_cmds = self.isolate_set_cmds(cmds)
        # set_callback_func = self.get_program_to_callback(BALL_MISS_PADDLE, set_cmds, False)
        # self.handler_dict[BALL_MISS_PADDLE].begin = set_callback_func

        callback_func = self.get_program_to_callback(BALL_HIT_WALL, cmds, bounce_or_not=False)
        self.handler_dict[BALL_MISS_PADDLE].separate = self.wrap(callback_func, self.ball_through_and_remove())

        cmds = self.program[BALL_IN_GOAL]
        if BOUNCE_BALL in cmds:
            # no need to remove after separate
            cmds = self.remove_bounce(cmds)
            callback_func = self.get_program_to_callback(BALL_IN_GOAL, cmds, bounce_or_not=True)

            self.handler_dict[BALL_IN_GOAL].begin = self.get_empty_func(True)
            self.handler_dict[BALL_IN_GOAL].post_solve = self.normalize_velocity()
            self.handler_dict[BALL_IN_GOAL].separate = callback_func  # self.get_empty_func()
        else:
            # need to remove
            callback_func = self.get_program_to_callback(BALL_IN_GOAL, cmds, bounce_or_not=False)
            self.handler_dict[BALL_IN_GOAL].begin = self.get_empty_func(False)
            self.handler_dict[BALL_IN_GOAL].separate = self.wrap(callback_func, self.ball_through_and_remove())

        # HIT paddle is not in charge of any ball missing
        cmds = self.program[BALL_HIT_PADDLE]
        cmds, set_cmds = self.isolate_set_cmds(cmds)
        if BOUNCE_BALL in cmds:
            # if bounces, then we don't need fall_through_and_remove
            # so everything else is fine
            cmds = self.remove_bounce(cmds)
            callback_func = self.get_program_to_callback(BALL_HIT_PADDLE, cmds, bounce_or_not=True)

            self.handler_dict[BALL_HIT_PADDLE].begin = callback_func
            self.handler_dict[BALL_HIT_PADDLE].post_solve = self.normalize_velocity()
            # self.handler_dict[BALL_HIT_PADDLE].separate = self.get_empty_func()

            set_callback_func = self.get_program_to_callback(BALL_HIT_PADDLE, set_cmds, True)
            self.handler_dict[BALL_HIT_PADDLE].separate = set_callback_func
        else:
            callback_func = self.get_program_to_callback(BALL_HIT_PADDLE, cmds, bounce_or_not=False)
            self.handler_dict[BALL_HIT_PADDLE].begin = callback_func


class RNG(object):
    def __init__(self):
        self.np_random = None
        self.curr_seed = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.curr_seed = seed
        return [seed]

    def choice(self, a, size=None, replace=True, p=None):
        return self.np_random.choice(a, size, replace, p)

    def randint(self, low, high=None, size=None, dtype=int):
        return self.np_random.randint(low, high, size, dtype)


class Bounce(object):
    """
    Collision handler will go in here

    Handles 3 situations:
    #1: when run
    #2: when left arrow
    #3: when right arrow

    """

    def __init__(self, program):
        pygame.init()

        self.rng = RNG()
        # I think it's this seed's problem
        self.seed()

        self.program = program

        self.all_sprites = pygame.sprite.Group()
        self.space = pymunk.Space()

        # if not initialized here, .convert() won't work
        self.screen = pygame.display.set_mode((screen_width, screen_height), 0, 32)  #
        self.clock = pygame.time.Clock()

        self.bg = Background()
        self.paddle = Paddle(self.space, self.all_sprites)

        self.walls = WallGroup(self.space, self.all_sprites)

        self.bottom = Bottom(self.space, self.all_sprites)
        self.goal = Goal(self.space, self.all_sprites)

        self.ball_group = BallGroup(self.space, self.all_sprites, self.rng)

        self.score_board = ScoreBoard()

        self.all_objs = [self.paddle, self.ball_group]

        self.engine = ShadowEngine(self.ball_group, self.paddle, self.goal, self.bottom, self.bg, self.walls,
                                   self.score_board, self.rng)
        self.cc = CollisionController(self.space, program, self.ball_group, self.engine, self.rng)

        self.cc.compile()

        self.action_cmds = [pygame.K_RIGHT, pygame.K_LEFT, PAUSE]  # "None" action correspond to pause

        self.fresh_run = True

    def seed(self, seed=None):
        # we use a class object, so that if we update seed here, it broadcasts into everywhere
        return self.rng.seed(seed)
        # self.np_random, seed = seeding.np_random(seed)
        # but the random seed in all other parts of the game is NOT necessarily updated
        # this means we need to do that, explicitly
        # return [seed]

    def sync_sprite(self):
        for o in self.all_objs:
            o.sync_sprite()

    def when_run_execute(self):
        cmds = self.program[WHEN_RUN]
        cmds = self.cc.remove_bounce(cmds)
        for cmd in cmds:
            self.engine.execute(cmd)

    def when_left_arrow(self, keys):
        return keys[pygame.K_LEFT]

    def when_right_arrow(self, keys):
        return keys[pygame.K_RIGHT]

    def run(self, debug=False):

        draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.when_run_execute()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            self.paddle.stop_moving()  # always stop paddle running at first

            keys = pygame.key.get_pressed()

            if self.when_left_arrow(keys):
                cmds = self.cc.remove_bounce(self.program[WHEN_LEFT_ARROW])
                list(map(lambda c: self.engine.execute(c), cmds))

            if self.when_right_arrow(keys):
                cmds = self.cc.remove_bounce(self.program[WHEN_RIGHT_ARROW])
                list(map(lambda c: self.engine.execute(c), cmds))

            # after a series of update...
            self.bg.draw(self.screen)

            # draw PyMunk objects (don't need this when actually playing)
            if debug:
                app.space.debug_draw(draw_options)

            self.sync_sprite()
            self.all_sprites.draw(self.screen)

            self.clock.tick(fps)
            self.space.step(1 / fps)

            self.score_board.draw(self.screen)

            pygame.display.flip()

    def prefill_keys(self):
        # used for agents
        keys = {}
        for k in self.action_cmds:
            keys[k] = False
        return keys

    def act(self, action):
        assert action in self.action_cmds, "need to supply correct action command: {}".format(self.action_cmds)

        # probably execute this at creation or at reset...not inside here
        if self.fresh_run:
            self.when_run_execute()
            self.fresh_run = False

        self.paddle.stop_moving()  # always stop paddle running at first

        keys = self.prefill_keys()
        keys[action] = True

        if self.when_left_arrow(keys):
            cmds = self.cc.remove_bounce(self.program[WHEN_LEFT_ARROW])
            list(map(lambda c: self.engine.execute(c), cmds))

        if self.when_right_arrow(keys):
            cmds = self.cc.remove_bounce(self.program[WHEN_RIGHT_ARROW])
            list(map(lambda c: self.engine.execute(c), cmds))

        # after a series of update...
        self.bg.draw(self.screen)
        self.score_board.draw(self.screen) # for RL, no display of scoreboard

        # draw PyMunk objects (don't need this when actually playing)
        # app.space.debug_draw(draw_options)

        self.sync_sprite()
        self.all_sprites.draw(self.screen)

        # Note: we no longer need to sleep/wait for the actual world clock
        # self.clock.tick(fps)
        self.space.step(1 / fps)

        pygame.display.flip()


if __name__ == '__main__':
    program = Program()

    program.set_correct()
    # program.set_correct_with_theme()
    # program.load("./bounce_programs/change_scene.json")
    # program.load("./bounce_programs/demo1.json")
    # program.load("./bounce_programs/easier_demo2.json")
    # program.load("./bounce_programs/speed_test.json")
    # program.load("./bounce_programs/mixed_theme_train.json")
    # program.load("./bounce_programs/empty.json")
    # program.load("./bounce_programs/broken_small/wall_not_bounce.json")
    app = Bounce(program)
    # app.seed(2222)
    # app.ball_group.get_new_ball_init()
    # print("assignment counter:", app.ball_group.assignment_counter)
    # print("Ball positions:", app.ball_group.ball_inits[:8])
    app.run()
