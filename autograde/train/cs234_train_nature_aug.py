import gym
import numpy as np
from collections import deque
from gym import spaces
from gym.wrappers import TimeLimit
import tensorflow as tf

import os
import sys
from autograde.train.cs234_utils import ReplayBuffer, Progbar, get_logger, export_plot
from autograde.rl_envs.bounce_env import BouncePixelEnv, Program, ONLY_SELF_SCORE, SELF_MINUS_HALF_OPPO

import pyglet

"""
Augmented to add human reocrded play 
"""


class SimpleImageViewer(object):
    """
    Modified version of gym viewer to chose format (RBG or I)
    see source here https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
    """

    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        ##########################
        ####### old version ######
        # assert arr.shape == (self.height, self.width, I), "You passed in an image with the wrong number shape"
        # image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes())
        ##########################

        ##########################
        ####### new version ######
        nchannels = arr.shape[-1]
        if nchannels == 1:
            _format = "I"
        elif nchannels == 3:
            _format = "RGB"
        else:
            raise NotImplementedError
        image = pyglet.image.ImageData(self.width, self.height, _format, arr.tobytes())
        ##########################

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale

    (400, 400, 3)
    ValueError: cannot reshape array of size 480000 into shape (210,160,3)

    100800 (200x200x3)

    """
    # state = np.reshape(state, [210, 160, 3]).astype(np.float32)
    state = np.reshape(state, [400, 400, 3]).astype(np.float32)

    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # karpathy
    # state = state[35:195]  # crop
    # state = state[::2, ::2]  # downsample by factor of 2
    state = state[::4, ::4]  # downsample by factor of 4

    state = state[:, :, np.newaxis]

    return state.astype(np.uint8)


class MaxAndSkipEnv(gym.Wrapper):
    """
    Wrapper from Berkeley's Assignment
    Takes a max pool over the last n states
    """

    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreproWrapper(gym.Wrapper):
    """
    Wrapper for Pong to apply preprocessing
    Stores the state into variable self.obs
    """

    def __init__(self, env, prepro, shape, overwrite_render=True, high=255):
        """
        Args:
            env: (gym env)
            prepro: (function) to apply to a state for preprocessing
            shape: (list) shape of obs after prepro
            overwrite_render: (bool) if True, render is overwriten to vizualise effect of prepro
            grey_scale: (bool) if True, assume grey scale, else black and white
            high: (int) max value of state after prepro
        """
        super(PreproWrapper, self).__init__(env)
        self.overwrite_render = overwrite_render
        self.viewer = None
        self.prepro = prepro
        self.observation_space = spaces.Box(low=0, high=high, shape=shape, dtype=np.uint8)
        self.high = high

    def step(self, action):
        """
        Overwrites _step function from environment to apply preprocess
        """
        obs, reward, done, info = self.env.step(action)
        self.obs = self.prepro(obs)
        return self.obs, reward, done, info

    def reset(self):
        self.obs = self.prepro(self.env.reset())
        return self.obs

    def _render(self, mode='human', close=False):
        """
        Overwrite _render function to vizualize preprocessing
        """

        if self.overwrite_render:
            if close:
                if self.viewer is not None:
                    self.viewer.close()
                    self.viewer = None
                return
            img = self.obs
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = SimpleImageViewer()
                self.viewer.imshow(img)


class LinearSchedule(object):
    def __init__(self, eps_begin, eps_end, nsteps):
        """
        Args:
            eps_begin: initial exploration
            eps_end: end exploration
            nsteps: number of steps between the two values of eps
        """
        self.epsilon = eps_begin
        self.eps_begin = eps_begin
        self.eps_end = eps_end
        self.nsteps = nsteps

    def update(self, t):
        """
        Updates epsilon

        Args:
            t: int
                frame number
        """
        ##############################################################
        """
        TODO: modify self.epsilon such that 
			  it is a linear interpolation from self.eps_begin to 
			  self.eps_end as t goes from 0 to self.nsteps
			  For t > self.nsteps self.epsilon remains constant
        """
        ##############################################################
        ################ YOUR CODE HERE - 3-4 lines ##################

        if t <= self.nsteps:
            self.epsilon = np.linspace(self.eps_begin, self.eps_end, num=int(self.nsteps) + 1)[t]
        else:
            self.epsilon = self.eps_end

        ##############################################################
        ######################## END YOUR CODE ############## ########


class LinearExploration(LinearSchedule):
    def __init__(self, env, eps_begin, eps_end, nsteps):
        """
        Args:
            env: gym environment
            eps_begin: float
                initial exploration rate
            eps_end: float
                final exploration rate
            nsteps: int
                number of steps taken to linearly decay eps_begin to eps_end
        """
        self.env = env
        super(LinearExploration, self).__init__(eps_begin, eps_end, nsteps)

    def get_action(self, best_action):
        """
        Returns a random action with prob epsilon, otherwise returns the best_action

        Args:
            best_action: int
                best action according some policy
        Returns:
            an action
        """
        ##############################################################
        """
        TODO: with probability self.epsilon, return a random action
                else, return best_action

                you can access the environment via self.env

                you may use env.action_space.sample() to generate 
                a random action        
        """
        ##############################################################
        ################ YOUR CODE HERE - 4-5 lines ##################

        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return best_action

        ##############################################################
        ######################## END YOUR CODE #######################


class QN(object):
    """
    Abstract Class for implementing a Q Network
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # build model
        self.build()

    def build(self):
        """
        Build model
        """
        pass

    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)

    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass

    def initialize(self):
        """
        Initialize variables if necessary
        """
        pass

    def get_best_action(self, state):
        """
        Returns best action according to the network

        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def update_target_params(self):
        """
        Update params of Q' with params of Q
        """
        pass

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0

        self.eval_reward = -21.

    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q = np.mean(max_q_values)
        self.avg_q = np.mean(q_values)
        self.std_q = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(int(t))
                    lr_schedule.update(int(t))
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                                  ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_eval), ("Max_Q", self.max_q),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                               self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()

        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval

    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # count reward
                total_reward += reward
                if done:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))

        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
            self.logger.info(msg)

        return avg_reward

    def record(self):
        """
        Re create an env and record a video for one episode
        """
        # env = gym.make(self.config.env_name)
        program = Program()
        program.set_correct()
        env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, reward_shaping=False)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(100, 100, 1),  # (80, 80, 1),
                            overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)

    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()


class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """

    def add_placeholders_op(self):
        raise NotImplementedError

    def get_q_values_op(self, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically
        to copy Q network to target Q network

        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError

    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError

    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError

    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state

    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg_Q", self.avg_q_placeholder)
        tf.summary.scalar("Max_Q", self.max_q_placeholder)
        tf.summary.scalar("Std_Q", self.std_q_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)

        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 self.sess.graph)

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)

    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """

        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size)

        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch,
            self.done_mask: done_mask_batch,
            self.lr: lr,
            # extra info
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm,
                                                               self.merged, self.train_op], feed_dict=fd)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


class Linear(DQN):
    """
    Implement Fully Connected with Tensorflow
    """

    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # this information might be useful
        state_shape = list(self.env.observation_space.shape)

        ##############################################################
        """
        TODO: 
            Add placeholders:
            Remember that we stack 4 consecutive frames together.
                - self.s: batch of states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.a: batch of actions, type = int32
                    shape = (batch_size)
                - self.r: batch of rewards, type = float32
                    shape = (batch_size)
                - self.sp: batch of next states, type = uint8
                    shape = (batch_size, img height, img width, nchannels x config.state_history)
                - self.done_mask: batch of done, type = bool
                    shape = (batch_size)
                - self.lr: learning rate, type = float32

        (Don't change the variable names!)

        HINT: 
            Variables from config are accessible with self.config.variable_name.
            Check the use of None in the dimension for tensorflow placeholders.
            You can also use the state_shape computed above.
        """
        ##############################################################
        ################YOUR CODE HERE (6-15 lines) ##################

        h, w, channels = state_shape

        self.s = tf.placeholder(tf.uint8, [None, h, w, channels * self.config.state_history])
        self.a = tf.placeholder(tf.int32, [None])
        self.r = tf.placeholder(tf.float32, [None])
        self.sp = tf.placeholder(tf.uint8, [None, h, w, channels * self.config.state_history])
        self.done_mask = tf.placeholder(tf.bool, [None])
        self.lr = tf.placeholder(tf.float32)

        ##############################################################
        ######################## END YOUR CODE #######################

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            Implement a fully connected with no hidden layer (linear
            approximation with bias) using tensorflow.

        HINT: 
            - You may find the following functions useful:
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse
        """
        ##############################################################
        ################ YOUR CODE HERE - 2-3 lines ##################

        with tf.variable_scope(scope):
            inputs = tf.layers.flatten(state)
            out = tf.layers.dense(inputs, num_actions, reuse=reuse)

        ##############################################################
        ######################## END YOUR CODE #######################

        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically
        to copy Q network weights to target Q network

        Remember that in DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes. If you're not familiar with the scope mechanism
        in tensorflow, read the docs
        https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope

        Periodically, we need to update all the weights of the Q network
        and assign them with the values from the regular network.
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        ##############################################################
        """
        TODO: 
            Add an operator self.update_target_op that for each variable in
            tf.GraphKeys.GLOBAL_VARIABLES that is in q_scope, assigns its
            value to the corresponding variable in target_q_scope

        HINT: 
            You may find the following functions useful:
                - tf.get_collection
                - tf.assign
                - tf.group (the * operator can be used to unpack a list)

        (be sure that you set self.update_target_op)
        """
        ##############################################################
        ################### YOUR CODE HERE - 5-10 lines #############
        q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=q_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_q_scope)
        assign_ops = [tf.assign(target_var, q_var) for q_var, target_var in zip(q_vars, target_vars)]
        self.update_target_op = tf.group(*assign_ops)

        ##############################################################
        ######################## END YOUR CODE #######################

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        # you may need this variable
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: 
            The loss for an example is defined as:
                Q_samp(s) = r if done
                          = r + gamma * max_a' Q_target(s', a')
                loss = (Q_samp(s) - Q(s, a))^2 
        HINT: 
            - Config variables are accessible through self.config
            - You can access placeholders like self.a (for actions)
                self.r (rewards) or self.done_mask for instance
            - You may find the following functions useful
                - tf.cast
                - tf.reduce_max
                - tf.reduce_sum
                - tf.one_hot
                - tf.squared_difference
                - tf.reduce_mean
        """
        ##############################################################
        ##################### YOUR CODE HERE - 4-5 lines #############

        # I can just multiply done_masks (using tf.cast)
        # this is a vector of states
        Q_samp = self.r + (self.config.gamma * tf.reduce_max(target_q, axis=1)) * (
                1 - tf.cast(self.done_mask, tf.float32))
        Q = tf.reduce_sum(q * tf.one_hot(self.a, num_actions), axis=1)
        self.loss = tf.reduce_mean(tf.squared_difference(Q_samp, Q))

        ##############################################################
        ######################## END YOUR CODE #######################

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm

        Args:
            scope: (string) name of the scope whose variables we are
                   differentiating with respect to
        """

        ##############################################################
        """
        TODO: 
            1. get Adam Optimizer
            2. compute grads with respect to variables in scope for self.loss
            3. if self.config.grad_clip is True, then clip the grads
                by norm using self.config.clip_val 
            4. apply the gradients and store the train op in self.train_op
                (sess.run(train_op) must update the variables)
            5. compute the global norm of the gradients (which are not None) and store 
                this scalar in self.grad_norm

        HINT: you may find the following functions useful
            - tf.get_collection
            - optimizer.compute_gradients
            - tf.clip_by_norm
            - optimizer.apply_gradients
            - tf.global_norm

             you can access config variables by writing self.config.variable_name
        """
        ##############################################################
        #################### YOUR CODE HERE - 8-12 lines #############

        optimizer = tf.train.AdamOptimizer(self.lr)
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        grad_vars = optimizer.compute_gradients(self.loss, vars)
        if self.config.clip_val:
            grad_vars = [(tf.clip_by_norm(g, self.config.clip_val), v) for g, v in grad_vars if g is not None]
        self.train_op = optimizer.apply_gradients(grad_vars)
        self.grad_norm = tf.global_norm([g for g, _ in grad_vars])

        ##############################################################
        ######################## END YOUR CODE #######################


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor)
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################

        with tf.variable_scope(scope, reuse=reuse):
            # first hidden layer is 8x8 16 filters with stride 4
            # then with ReLU nonlinearity
            inputs = tf.layers.conv2d(state, kernel_size=(8, 8), filters=32, strides=4, activation=tf.nn.relu,
                                      padding="same")
            # second layer convolves 32 4 × 4 filters with stride 2, ReLU
            inputs = tf.layers.conv2d(inputs, kernel_size=(4, 4), filters=64, strides=2, activation=tf.nn.relu,
                                      padding="same")
            # by a third convolutional layer that convolves 64 filters of 3x3 with
            # stride 1 followed by a rectifier
            inputs = tf.layers.conv2d(inputs, kernel_size=(4, 4), filters=64, strides=1, activation=tf.nn.relu,
                                      padding="same")

            # The final hidden layer is fully-connected and consists of 256 rectifier units
            inputs = tf.layers.flatten(inputs)
            inputs = tf.layers.dense(inputs, 512, activation=tf.nn.relu)

            out = tf.layers.dense(inputs, num_actions)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


class NatureQNWithHuman(NatureQN):
    """
    Override some key aspects for executing human policy
    We only override training

    .run() calls .train()
    """

    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # Every freq, we sample ONE human play recording
        # and execute it
        human_play_freq = self.config.human_play_freq

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = last_human_play = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self.evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            total_reward = 0  # episodic reward
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                last_human_play += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(int(t))
                    lr_schedule.update(int(t))
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                                  ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_eval), ("Max_Q", self.max_q),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t,
                                                                               self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            # add human play
            if (t > self.config.learning_start) and (last_human_play > self.config.human_play_freq):
                last_human_play = 0
                # we masquerade human play execution as "training"
                # we first take the human actions out!
                sampled_idx = np.random.randint(0, len(self.config.human_play_file_names))
                file_name, seed = self.config.human_play_file_names[sampled_idx]
                human_actions = np.load(self.config.human_play_dir + file_name)['frames']

                print("")
                print("Performing human policy idx {}".format(sampled_idx))

                # now execute this!
                state = self.env.reset()
                self.env.seed(seed)

                action_idx = 0
                total_reward = 0
                while True:

                    # replay memory stuff
                    idx = replay_buffer.store_frame(state)
                    # q_input = replay_buffer.encode_recent_observation()

                    # choose an action according to human policy
                    action = human_actions[action_idx]

                    # perform the action
                    new_state, reward, done, info = self.env.step(action)

                    # store the transition
                    replay_buffer.store_effect(idx, action, reward, done)
                    state = new_state

                    # we do not perform training steps in here

                    total_reward += reward

                    if done:
                        break

                    action_idx += 1

                print("Human policy total reward: {}\n".format(total_reward))

                rewards.append(total_reward)  # adding to total rewards

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self.evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)


class Config():
    # env config
    render_train = False
    render_test = False
    env_name = "Pong-v0"
    overwrite_render = True
    record = True
    high = 255.

    # output config
    output_path = "results/q5_train_atari_nature_one_ball/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path + "monitor/"

    # model and training config
    num_episodes_test = 20 # 50
    grad_clip = True
    clip_val = 10
    saving_freq = 250000
    log_freq = 50
    eval_freq = 250000
    record_freq = 250000
    soft_epsilon = 0.05

    # nature paper hyper params
    nsteps_train = 3000000
    batch_size = 32
    buffer_size = 1000000
    target_update_freq = 10000
    gamma = 0.99
    learning_freq = 4
    state_history = 4
    skip_frame = 2
    lr_begin = 0.00025
    lr_end = 0.00005
    lr_nsteps = nsteps_train / 2
    eps_begin = 1
    eps_end = 0.1
    eps_nsteps = 1500000  # 1000000 # it probably decays too quickly!
    learning_start = 20000 # 50000

    # human play guidance
    use_human_play = False
    human_play_freq = 50000
    human_play_dir = "./autograde/rl_envs/bounce_humanplay_recordings/"
    human_play_file_names = [("human_actions_2222_max_skip_2_converted.npz", 2222)]


def main(args):
    USE_HUMAN = args.human_play

    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    config = Config()
    config.use_human_play = USE_HUMAN

    # make env
    program = Program()
    program.set_correct()
    # program.set_correct_with_theme()

    env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, reward_shaping=False, num_ball_to_win=1,
                         finish_reward=0)
    env = MaxAndSkipEnv(env, skip=2)  # maybe 2 is better?
    env = PreproWrapper(env, prepro=greyscale, shape=(100, 100, 1),  # (80, 80, 1),
                        overwrite_render=True)

    env = TimeLimit(env, max_episode_steps=500) # 1500

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps)

    # train model
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with tf.Session(config=tf_config):
        if USE_HUMAN:
            model = NatureQNWithHuman(env, config)
        else:
            model = NatureQN(env, config)
        model.run(exp_schedule, lr_schedule)


def replay_human_play_with_gym_wrapper(human_play_npz, seed, max_len=1500, max_skip=2):
    import os
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
    os.environ['SDL_AUDIODRIVER'] = 'dsp'

    # we won't show videos, because it's harder.
    # We just want to prove that we get the reward
    program = Program()
    program.set_correct()
    # program.set_correct_with_theme()

    env = BouncePixelEnv(program, SELF_MINUS_HALF_OPPO, reward_shaping=False)
    env = MaxAndSkipEnv(env, skip=2)  # maybe 2 is better?
    env = PreproWrapper(env, prepro=greyscale, shape=(100, 100, 1),  # (80, 80, 1),
                        overwrite_render=True)

    # before MaxSkip, we can do 3000
    # since MaxSkip does 2 actions at once, we half it to 1500
    env = TimeLimit(env, max_episode_steps=max_len)

    human_actions = np.load("./autograde/rl_envs/bounce_humanplay_recordings/" + human_play_npz)['frames']

    print("number of human actions: {}".format(human_actions.shape[0]))

    obs = env.reset()
    env.seed(seed)
    for i in range(max_len):
        # action = np.random.randint(env.action_space.n, size=1)
        action = human_actions[i]
        obs, rewards, dones, info = env.step(action)
        if rewards != 0:
            print(rewards)
        if dones:
            break


if __name__ == '__main__':
    # test if human play can be loaded and played correctly
    # This works!!!!
    # replay_human_play_with_gym_wrapper("human_actions_2222_max_skip_2_converted.npz", seed=2222, max_skip=2)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--human_play", action="store_true", help="Execute human play every N iterations")

    args = parser.parse_args()

    main(args)
