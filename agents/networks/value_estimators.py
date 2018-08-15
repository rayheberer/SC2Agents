"""Neural networks that output value estimates for actions, given a state."""

import numpy as np
import tensorflow as tf


class PlayerRelativeMovementCNN(object):
    """Uses feature_screen.player_relative to assign q value to movements."""

    def __init__(self,
                 spatial_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name="DQN"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.spatial_dimensions = spatial_dimensions
        self.learning_rate = learning_rate
        self.name = name
        self.save_path = save_path

        # build graph
        self._build()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Loss", self.loss)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Batch_Max_Q", self.max_q)
            tf.summary.scalar("Batch_Mean_Q", self.mean_q)
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

    def write_summary(self, sess, states, actions, targets, score):
        """Write summary to Tensorboard."""
        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets,
                       self.score: score})
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def optimizer_op(self, sess, states, actions, targets):
        """Perform one iteration of gradient updates."""
        loss, _ = sess.run(
            [self.loss, self.optimizer],
            feed_dict={self.inputs: states,
                       self.actions: actions,
                       self.targets: targets})

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def _build(self):
        """Construct graph."""
        with tf.variable_scope(self.name):
            # score tracker
            self.score = tf.placeholder(
                tf.int32,
                [],
                name="score")

            # global step trackers for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name="global_step")

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name="global_episode")

            # placeholders
            self.inputs = tf.placeholder(
                tf.int32,
                [None, *self.spatial_dimensions],
                name="inputs")

            self.actions = tf.placeholder(
                tf.float32,
                [None, np.prod(self.spatial_dimensions)],
                name="actions")

            self.targets = tf.placeholder(
                tf.float32,
                [None],
                name="targets")

            self.increment_global_episode = tf.assign(
                self.global_episode,
                self.global_episode + 1,
                name="increment_global_episode")

            # spatial coordinates are given in y-major screen coordinate space
            # transpose them to (x, y) space before beginning
            self.transposed = tf.transpose(
                self.inputs,
                perm=[0, 2, 1],
                name="transpose")

            # embed layer (one-hot in channel dimension, 1x1 convolution)
            # the player_relative feature layer has 5 categorical values
            self.one_hot = tf.one_hot(
                self.transposed,
                depth=5,
                axis=-1,
                name="one_hot")

            self.embed = tf.layers.conv2d(
                inputs=self.one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="embed")

            # convolutional layer
            self.conv1 = tf.layers.conv2d(
                inputs=self.embed,
                filters=16,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="SAME",
                name="conv1")

            self.conv1_activation = tf.nn.relu(
                self.conv1,
                name="conv1_activation")

            # spatial output layer
            self.output = tf.layers.conv2d(
                inputs=self.conv1_activation,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME",
                name="output")

            self.flatten = tf.layers.flatten(self.output, name="flat")

            # value estimate trackers for summaries
            self.max_q = tf.reduce_max(self.flatten, name="max")
            self.mean_q = tf.reduce_mean(self.flatten, name="mean")

            # optimization: MSE between state predicted Q and target Q
            self.prediction = tf.reduce_sum(
                tf.multiply(self.flatten, self.actions),
                axis=1,
                name="prediction")

            self.loss = tf.reduce_mean(
                tf.square(self.targets - self.prediction),
                name="loss")

            self.optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss,
                                             global_step=self.global_step)
