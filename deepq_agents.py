"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# neural network hyperparameters
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
flags.DEFINE_float("discount_factor", 0.9, "Discount factor.")

# agent hyperparameters
flags.DEFINE_float("epsilon_max", 1, "Initial exploration probability.")
flags.DEFINE_float("epsilon_min", 0.1, "Final exploration probability.")
flags.DEFINE_integer("epsilon_decay_steps", 2000000, "Steps for linear decay.")
flags.DEFINE_integer("train_every", 1, "Steps between training batches.")

flags.DEFINE_integer("max_memory", 1024, "Experience Replay buffer size.")
flags.DEFINE_integer("batch_size", 8, "Training batch size.")

# run settings
flags.DEFINE_string(
    "save_dir",
    "./checkpoints/",
    "Model checkpoint save directory.")
flags.DEFINE_string(
    "summary_path",
    "./tensorboard/deepq/",
    "Tensorboard summary write path.")

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS


class Memory(object):
    """Memory buffer for Experience Replay."""

    def __init__(self, max_size):
        """Initialize a buffer containing max_size experiences."""
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences from the buffer."""
        buffer_size = len(self.buffer)
        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False)

        return [self.buffer[i] for i in index]

    def __len__(self):
        """Interface to access buffer length."""
        return len(self.buffer)


class DQNMoveOnly(base_agent.BaseAgent):
    """A DQN that receives `player_relative` features and takes movements."""

    def __init__(self):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNMoveOnly, self).__init__()

        # hyperparameters TODO: set these using flags
        self.learning_rate = FLAGS.learning_rate
        self.discount_factor = FLAGS.discount_factor
        self.epsilon_max = FLAGS.epsilon_max
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon_decay_steps = FLAGS.epsilon_decay_steps

        self.train_every = FLAGS.train_every

        # build network
        print("Building model...")
        tf.reset_default_graph()
        self._build_network()
        print("Done.")

        # initialize Experience Replay memory buffer
        self.memory = Memory(FLAGS.max_memory)
        self.batch_size = FLAGS.batch_size

        self.last_state = None
        self.last_action = None

        # setup summary writer
        self.writer = tf.summary.FileWriter(FLAGS.summary_path)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Score", self.score)
        self.write_op = tf.summary.merge_all()

        # setup model saver
        self.saver = tf.train.Saver()
        self.save_path = FLAGS.save_dir + "DQNPlayerRelativeMoveScreen.ckpt"

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.saver.restore(self.sess, self.save_path)
        else:
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.sess.run(self.increment_global_episode)
        score = self.reward
        self.reward = 0

        self.last_state = None
        self.last_action = None

        global_episodes = self.global_episodes.eval(session=self.sess)

        # don't do anything else for 1st episode
        if self.episodes > 1:

            # save current model
            self.saver.save(self.sess, self.save_path)
            print("Model Saved")

            # write summaries from last episode
            states, actions, targets = self._get_batch()
            summary = self.sess.run(
                self.write_op,
                feed_dict={self.inputs: states,
                           self.actions: actions,
                           self.targets: targets,
                           self.score: score})

            self.writer.add_summary(summary, global_episodes - 1)
            self.writer.flush()
            print("Summary Written")

        print("Starting Global Episode", global_episodes)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # predict an action to take and take it
            state = obs.observation.feature_screen.player_relative
            x, y = self._epsilon_greedy_action_selection(state)

            if (self.steps % self.train_every == 0 and
                    len(self.memory) > self.batch_size):
                self._train_network()

            if self.last_state is not None:
                self.memory.add(
                    (self.last_state,
                     self.last_action,
                     obs.reward,
                     state))

            self.last_state = state
            self.last_action = np.ravel_multi_index(
                (x, y),
                feature_screen_size)

            return FUNCTIONS.Move_screen("now", (x, y))
        else:
            return FUNCTIONS.select_army("select")

    def _epsilon_greedy_action_selection(self, state):
        """Choose action from state with epsilon greedy strategy."""
        step = self.global_step.eval(session=self.sess)
        epsilon = max(
            self.epsilon_min,
            (self.epsilon_max - ((self.epsilon_max - self.epsilon_min) *
                                 step / self.epsilon_decay_steps)))

        if epsilon > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])

            return x, y

        else:
            inputs = np.expand_dims(state, 0)

            q_values = self.sess.run(
                self.output,
                feed_dict={self.inputs: inputs})

            max_index = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y

    def _train_network(self):
        states, actions, targets = self._get_batch()

        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

    def _get_batch(self):
        batch = self.memory.sample(self.batch_size)
        states = np.array([each[0] for each in batch])
        actions = np.array([each[1] for each in batch])
        rewards = np.array([each[2] for each in batch])
        next_states = np.array([each[3] for each in batch])

        # one-hot encode actions
        actions = np.eye(np.prod(feature_screen_size))[actions]

        # get targets
        next_outputs = self.sess.run(
            self.output,
            feed_dict={self.inputs: next_states})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets

    def _build_network(self):
        """CNN used to approximate Q table with screen features."""
        with tf.variable_scope('DQN'):

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, *feature_screen_size],
                name='inputs')

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(feature_screen_size)],
                name='actions')

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name='targets')

            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name='score')

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step')

            self.global_episodes = tf.Variable(
                0,
                trainable=False,
                name='global_episodes')

            self.increment_global_episode = tf.assign(self.global_episodes,
                                                      self.global_episodes + 1)

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.inputs,
                depth=5,
                axis=-1,
                name='one_hot')

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding='VALID',
                name='embed')

            # convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=64,
                kernel_size=[3, 3],
                strides=[2, 2],
                padding='SAME',
                name='conv1')

            self.conv1_activation = tf.nn.elu(
                self.conv1,
                name='conv1_activation')

            # output layers
            self.flatten = tf.layers.flatten(self.conv1_activation)

            self.dense = tf.layers.dense(
                inputs=self.flatten,
                units=256,
                activation=tf.nn.elu,
                name='fully_connected')

            # output shape: (1, feature_screen_size[0]*feature_screen_size[1])
            self.output = tf.layers.dense(
                inputs=self.dense,
                units=np.prod(feature_screen_size),
                activation=None,
                name='output')

            # optimization: RMSE between state predicted Q and target Q
            self.predicted_Q = tf.reduce_sum(
                tf.multiply(self.output, self.actions),
                axis=1)

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.predicted_Q),
                name='loss')

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)
