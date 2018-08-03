"""Deep Q-learning agents."""
import numpy as np
import tensorflow as tf

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions

FLAGS = flags.FLAGS
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

FUNCTIONS = actions.FUNCTIONS


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


class DQNPlayerRelativeMoveScreen(base_agent.BaseAgent):
    """A DQN that receives `player_relative` features and takes movements."""

    def __init__(self):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNPlayerRelativeMoveScreen, self).__init__()
        self.episode_steps = 0

        # hyperparameters TODO: set these using flags
        self.learning_rate = 0.0001  # larger learning rates explode gradients
        self.discount_factor = 0.9
        self.epsilon = 1
        self.epsilon_step_decay_amount = 0.01
        self.epsilon_episode_decay_factor = 0.9

        # build network, and initialize session
        self._build_network()

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        # initialize Experience Replay memory buffer
        self.memory = Memory(1024)
        self.batch_size = 16

        self.last_state = None
        self.last_action = None

    def reset(self):
        """Handle the beginning of new episodes."""
        super(DQNPlayerRelativeMoveScreen, self).reset()
        self.episode_steps = 0

        self.last_state = None
        self.last_action = None

        self.epsilon *= self.epsilon_episode_decay_factor**self.episodes

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        super(DQNPlayerRelativeMoveScreen, self).step(obs)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            # predict an action to take and take it
            state = obs.observation.feature_screen.player_relative
            x, y = self._epsilon_greedy_action_selection(state)

            if len(self.memory) > self.batch_size:
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
                strides=[1, 1],
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
                tf.square(self.targets - self.predicted_Q))

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)

    def _epsilon_greedy_action_selection(self, state):
        """Choose action from state with epsilon greedy strategy."""
        explore_probability = self.epsilon - (self.epsilon_step_decay_amount *
                                              self.episode_steps)

        if explore_probability > np.random.rand():
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

        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

        print(loss)
