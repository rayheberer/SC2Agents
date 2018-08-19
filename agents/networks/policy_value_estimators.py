"""Neural networks that output both value and optimal policy estimations."""
import tensorflow as tf

from agents.networks.preprocessing import preprocess_spatial_features

from pysc2.lib import actions, features

SCREEN_FEATURES = features.SCREEN_FEATURES
MINIMAP_FEATURES = features.MINIMAP_FEATURES

NUM_ACTIONS = len(actions.FUNCTIONS)

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [actions.TYPES[0], actions.TYPES[2]]
MINIMAP_TYPES = [actions.TYPES[1]]


class AtariNet(object):
    """Estimates value and policy with shared parameters."""

    def __init__(self,
                 screen_dimensions,
                 minimap_dimensions,
                 learning_rate,
                 value_gradient_strength,
                 regularization_strength,
                 save_path=None,
                 summary_path=None,
                 name="AtariNet"):
        """Initialize instance-specific hyperparameters, build tf graph."""
        self.screen_dimensions = screen_dimensions
        self.minimap_dimensions = minimap_dimensions
        self.learning_rate = learning_rate
        self.value_gradient_strength = value_gradient_strength
        self.regularization_strength = regularization_strength
        self.save_path = save_path

        # build graph
        with tf.variable_scope(name):
            self._build()
            self._build_optimization()

        # setup summary writer
        if summary_path:
            self.writer = tf.summary.FileWriter(summary_path)
            tf.summary.scalar("Score", self.score)
            tf.summary.scalar("Policy_Loss", self.policy_gradient)
            tf.summary.scalar("Value_Loss", self.value_gradient)
            tf.summary.scalar("Entropy", self.entropy)
            tf.summary.scalar("A2C_Loss", self.a2c_gradient)
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

    def write_summary(self, sess, score, feed_dict):
        """Write summary to Tensorboard."""
        feed_dict[self.score] = score

        global_episode = self.global_episode.eval(session=sess)
        summary = sess.run(
            self.write_op,
            feed_dict=feed_dict)
        self.writer.add_summary(summary, global_episode - 1)
        self.writer.flush

    def increment_global_episode_op(self, sess):
        """Increment the global episode tracker."""
        sess.run(self.increment_global_episode)

    def optimizer_op(self, sess, feed_dict):
        """Perform one iteration of gradient updates."""
        sess.run(self.optimizer, feed_dict=feed_dict)

    def _build(self):
        """Construct graph for state representation."""
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

        self.increment_global_episode = tf.assign(
            self.global_episode,
            self.global_episode + 1,
            name="increment_global_episode")

        # state placeholders
        self.screen_features = tf.placeholder(
            tf.int32,
            [None, len(SCREEN_FEATURES), *self.screen_dimensions],
            name="screen_features")

        self.minimap_features = tf.placeholder(
            tf.int32,
            [None, len(MINIMAP_FEATURES), *self.minimap_dimensions],
            name="minimap_features")

        self.flat_features = tf.placeholder(
            tf.float32,
            [None, len(features.Player)],
            name="flat_features")

        # preprocessing
        self.screen_processed = preprocess_spatial_features(
            self.screen_features,
            screen=True)

        self.minimap_processed = preprocess_spatial_features(
            self.minimap_features,
            screen=False)

        self.flat_processed = tf.log(
            self.flat_features + 1.,
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
        # action function argument placeholders (for optimization)
        self.argument_policy = dict()
        self.arguments = dict()
        for arg_type in actions.TYPES:

            # for spatial actions, represent each dimension independently
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    units = self.screen_dimensions
                elif arg_type in MINIMAP_TYPES:
                    units = self.minimap_dimensions

                arg_policy_x = tf.layers.dense(
                    inputs=self.state_representation,
                    units=units[0],
                    activation=tf.nn.softmax)

                arg_policy_y = tf.layers.dense(
                    inputs=self.state_representation,
                    units=units[1],
                    activation=tf.nn.softmax)

                self.argument_policy[str(arg_type) + "x"] = arg_policy_x
                self.argument_policy[str(arg_type) + "y"] = arg_policy_y

                arg_placeholder_x = tf.placeholder(
                    tf.float32,
                    shape=[None, units[0]])

                arg_placeholder_y = tf.placeholder(
                    tf.float32,
                    shape=[None, units[1]])

                self.arguments[str(arg_type) + "x"] = arg_placeholder_x
                self.arguments[str(arg_type) + "y"] = arg_placeholder_y

            else:
                arg_policy = tf.layers.dense(
                    inputs=self.state_representation,
                    units=arg_type.sizes[0],
                    activation=tf.nn.softmax)

                self.argument_policy[str(arg_type)] = arg_policy

                arg_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[None, arg_type.sizes[0]])

                self.arguments[str(arg_type)] = arg_placeholder

        # value estimation
        self.value_estimate = tf.layers.dense(
            inputs=self.state_representation,
            units=1,
            activation=None,
            name="value_estimate")

    def _build_optimization(self):
        """Construct graph for network updates."""
        # target placeholders
        self.actions = tf.placeholder(
            tf.float32,
            [None, NUM_ACTIONS],
            name="actions")

        self.returns = tf.placeholder(
            tf.float32,
            [None],
            name="returns")

        # compute advantage
        self.action_probability = tf.reduce_sum(
            self.function_policy * self.actions,
            axis=1,
            name="action_probability")

        self.args_probability = 1.
        for arg_type in self.arguments:
            arg_prob = tf.reduce_sum(
                self.arguments[arg_type] * self.argument_policy[arg_type])
            nonzero_probs = tf.cond(
                tf.logical_not(tf.equal(arg_prob, 0)),
                true_fn=lambda: arg_prob,
                false_fn=lambda: 1.)
            self.args_probability *= nonzero_probs

        self.advantage = tf.subtract(
            self.returns,
            tf.squeeze(tf.stop_gradient(self.value_estimate)),
            name="advantage")

        # a2c gradient = policy gradient + value gradient + regularization
        self.policy_gradient = -tf.reduce_mean(
            (self.advantage *
             tf.log(self.action_probability * self.args_probability)),
            name="policy_gradient")

        self.value_gradient = -tf.reduce_mean(
            self.advantage * tf.squeeze(self.value_estimate),
            name="value_gradient")

        # only including function identifier entropy, not args
        self.entropy = tf.reduce_sum(
            self.function_policy * tf.log(self.function_policy),
            name="entropy_regularization")

        self.a2c_gradient = tf.add_n(
            inputs=[self.policy_gradient,
                    self.value_gradient_strength * self.value_gradient,
                    self.regularization_strength * self.entropy],
            name="a2c_gradient")

        self.optimizer = tf.train.RMSPropOptimizer(
            self.learning_rate).minimize(self.a2c_gradient,
                                         global_step=self.global_step)
