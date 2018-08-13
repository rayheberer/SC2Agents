"""Neural networks that output both value and optimal policy estimations."""
import tensorflow as tf


class AtariNet(object):
    """Estimates value and policy with shared parameters."""

    def __init__(self,
                 screen_dimensions,
                 minimap_dimensions,
                 learning_rate,
                 save_path=None,
                 summary_path=None,
                 name='AtariNet'):
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
                name='score')

            # global step traker for multiple runs restoring from ckpt
            self.global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step')

            self.global_episode = tf.Variable(
                0,
                trainable=False,
                name='global_episode')

            # placeholders
            self.screen_features = tf.placeholder(
                tf.int32,
                [*self.screen_dimensions, 17],
                name='screen_features')

            self.minimap_features = tf.placeholder(
                tf.int32,
                [*self.minimap_dimensions, 7],
                name='minimap_features')

            self.nonspatial_features = tf.placeholder(
                tf.int32,
                [],
                name='nonspatial_features')

            # preprocessing

            # convolutional layers for screen features

            # convolutional layers for minimap features

            # linear layer for non-spatial features (tanh activation)

            # flatten and concatenate

            # linear layer with ReLU activation

            # action function identifier policy

            # action function argument policies (nonspatial)

            # action functio argument policies (x and y)

            # value estimation

            # optimization
