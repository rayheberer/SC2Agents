"""Deep Q-learning agents."""
import numpy as np
import tensorflow as tf

from absl import flags

from pysc2.agents import base_agent
from pysc2.lib import actions

FLAGS = flags.FLAGS
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

FUNCTIONS = actions.FUNCTIONS


class DQNPlayerRelativeMoveScreen(base_agent.BaseAgent):
    """A DQN that receives `player_relative` features and takes movements."""

    def __init__(self):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNPlayerRelativeMoveScreen, self).__init__()

        self._build_network()
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        super(DQNPlayerRelativeMoveScreen, self).step(obs)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            x, y = self._epsilon_greedy_action_selection(obs)
            print(obs.reward)
            return FUNCTIONS.Move_screen("now", (x, y))
        else:
            return FUNCTIONS.select_army("select")

    def _epsilon_greedy_action_selection(self, obs, decay_rate=0.1):
        """Choose action from state with epsilon greedy strategy."""
        explore_probability = np.exp(-decay_rate * self.episodes)

        if explore_probability > np.random.rand():
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])

            return x, y

        else:
            inputs = obs.observation.feature_screen.player_relative

            q_values = self.sess.run(
                self.output,
                feed_dict={self.inputs: np.expand_dims(inputs, 0)}
            )

            max_index = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y

    def _build_network(self):
        """CNN used to approximate Q table with screen features."""
        with tf.variable_scope('DQN'):

            # placeholders
            self.inputs = tf.placeholder(tf.int32,
                                         [None, *feature_screen_size],
                                         name='inputs')

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
                name='embed'
            )

            # first convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding='SAME',
                name='conv1'
            )
            self.conv1_batchnorm = tf.layers.batch_normalization(
                self.conv1,
                training=True,
                name='conv1_batchnorm'
            )
            self.conv1_activation = tf.nn.elu(
                self.conv1_batchnorm,
                name='conv1_activation'
            )

            # second convolutional layer
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='SAME',
                name='conv2'
            )
            self.conv2_batchnorm = tf.layers.batch_normalization(
                self.conv2,
                training=True,
                name='conv2_batchnorm'
            )
            self.conv2_activation = tf.nn.elu(
                self.conv2_batchnorm,
                name='conv2_activation'
            )

            # third convolutional layer
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2_activation,
                filters=128,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding='SAME',
                name='conv3'
            )
            self.conv3_batchnorm = tf.layers.batch_normalization(
                self.conv3,
                training=True,
                name='conv3_batchnorm'
            )
            self.conv3_activation = tf.nn.elu(
                self.conv3_batchnorm,
                name='conv3_activation'
            )

            # output layers
            self.flatten = tf.layers.flatten(self.conv3_activation)

            self.dense = tf.layers.dense(
                inputs=self.flatten,
                units=512,
                activation=tf.nn.elu,
                name='fully_connected'
            )

            self.output = tf.layers.dense(
                inputs=self.dense,
                units=np.prod(feature_screen_size),
                activation=None,
                name='output')

            # output shape: (1, feature_screen_size[0]*feature_screen_size[1])
