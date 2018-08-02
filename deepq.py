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
            inputs = obs.observation.feature_screen.player_relative
            inputs = np.expand_dims(inputs, 0)
            test = self.sess.run(
                self.embed,
                feed_dict={self.inputs: inputs}
            )
            print(test)
            return FUNCTIONS.no_op()
        else:
            return FUNCTIONS.select_army("select")

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
