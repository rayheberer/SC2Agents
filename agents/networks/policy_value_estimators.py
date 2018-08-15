"""Neural networks that output both value and optimal policy estimations."""
import tensorflow as tf

import agents.networks.preprocessing as preprocessing

from pysc2.lib import actions, features

SCREEN_FEATURES = features.SCREEN_FEATURES
MINIMAP_FEATURES = features.MINIMAP_FEATURES

NUM_ACTIONS = len(actions.FUNCTIONS)


class AtariNet(object):
    """Estimates value and policy with shared parameters."""

    def __init__(self,
                 screen_dimensions,
                 minimap_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="AtariNet"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.screen_dimensions = screen_dimensions
        self.minimap_dimensions = minimap_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Score", self.score)
            self.write_op = tf.summary.merge_all()

        # setup model saver
        if self.save_path:
            self.saver = tf.train.Saver()

    def save_model(self, sess):
        """Write tensorflow ckpt."""
        self.saver.save(sess, self.save_path)

    def load(self, sess):
        """Restore from ckpt."""
        self.saver.restore(sess, self.save_path)

    def write_summary(self, sess, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def optimizer_op(self):
        """Perform one iteration of gradient updates."""
        raise NotImplementedError

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step traker for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # state placeholders
            self.screen_features = tf.placeholder(
                tf.int32,
                [len(SCREEN_FEATURES), *self.screen_dimensions],
                name="screen_features")

            self.minimap_features = tf.placeholder(
                tf.int32,
                [len(MINIMAP_FEATURES), *self.minimap_dimensions],
                name="minimap_features")

            self.flat_features = tf.placeholder(
                tf.float32,
                [len(features.Player)],
                name="flat_features")

            self.available_actions = tf.placeholder(
                tf.float32,
                [None],
                name="available_actions")

            # target placeholders

            # preprocessing (expand dims used because batch size is always 1)
            self.screen_processed = preprocessing.preprocess_spatial_features(
                self.screen_features,
                screen=True)

            self.screen_processed = tf.expand_dims(
                self.screen_processed,
                axis=0,
                name="screen_processed")

            self.minimap_processed = preprocessing.preprocess_spatial_features(
                self.minimap_features,
                screen=False)

            self.minimap_processed = tf.expand_dims(
                self.minimap_processed,
                axis=0,
                name="minimap_processed")

            self.flat_processed = tf.expand_dims(
                tf.log(self.flat_features + 1.),
                0,
                name="flat_processed")

            # convolutional layers for screen features
            self.screen_conv1 = tf.layers.conv2d(
                inputs=self.screen_processed,
                filters=16,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="SAME",
                name="screen_conv1")

            self.screen_activation1 = tf.nn.relu(
                self.screen_conv1,
                name="screen_activation1")

            self.screen_conv2 = tf.layers.conv2d(
                inputs=self.screen_activation1,
                filters=32,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="SAME",
                name="screen_conv2")

            self.screen_activation2 = tf.nn.relu(
                self.screen_conv2,
                name="screen_activation2")

            # convolutional layers for minimap features
            self.minimap_conv1 = tf.layers.conv2d(
                inputs=self.minimap_processed,
                filters=16,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="SAME",
                name="minimap_conv1")

            self.minimap_activation1 = tf.nn.relu(
                self.minimap_conv1,
                name="minimap_activation1")

            self.minimap_conv2 = tf.layers.conv2d(
                inputs=self.minimap_activation1,
                filters=32,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="SAME",
                name="minimap_conv2")

            self.minimap_activation2 = tf.nn.relu(
                self.minimap_conv2,
                name="minimap_activation2")

            # linear layer for non-spatial features (tanh activation)
            self.flat_linear = tf.layers.dense(
                inputs=self.flat_processed,
                units=64,
                activation=tf.tanh,
                name="flat_linear")

            # flatten and concatenate
            self.screen_flat = tf.layers.flatten(
                self.screen_activation2,
                name="screen_flat")

            self.minimap_flat = tf.layers.flatten(
                self.minimap_activation2,
                name="minimap_flat")

            self.concat = tf.concat(
                values=[self.screen_flat, self.minimap_flat, self.flat_linear],
                axis=1,
                name="concat")

            # linear layer with ReLU activation
            self.state_representation = tf.layers.dense(
                inputs=self.concat,
                units=256,
                activation=tf.nn.relu,
                name="state_representation")

            # action function identifier policy
            self.function_policy = tf.squeeze(tf.layers.dense(
                inputs=self.state_representation,
                units=NUM_ACTIONS,
                activation=tf.nn.softmax),
                name="function_policy")

            # action function argument policies (nonspatial)
            self.argument_policies = dict()
            for arg_type in actions.TYPES:

                # for spatial actions, represent each dimension independently
                for i in range(len(arg_type.sizes)):
                    arg_policy = tf.layers.dense(
                        inputs=self.state_representation,
                        units=arg_type.sizes[i],
                        activation=tf.nn.softmax)

                    self.argument_policies[str(arg_type) + str(i)] = arg_policy

            # value estimation
            self.value_estimate = tf.layers.dense(
                inputs=self.state_representation,
                units=1,
                activation=None,
                name="value_estimate")

            # optimization
