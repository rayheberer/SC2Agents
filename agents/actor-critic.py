"""Actor-critic agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.policy_value_estimators as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# pysc2 convenience
FUNCTIONS = sc2_actions.FUNCTIONS
FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
FunctionCall = sc2_actions.FunctionCall

# manually state the argument types which take points on screen/minimap
SCREEN_TYPES = [sc2_actions.TYPES[0], sc2_actions.TYPES[2]]
MINIMAP_TYPES = [sc2_actions.TYPES[1]]


class A2C(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 value_gradient_strength=FLAGS.value_gradient_strength,
                 regularization_strength=FLAGS.regularization_strength,
                 discount_factor=FLAGS.discount_factor,
                 trajectory_training_steps=FLAGS.trajectory_training_steps,
                 training=FLAGS.training,
                 save_dir="./checkpoints/",
                 ckpt_name="A2C",
                 summary_path="./tensorboard/A2C"):
        """Initialize rewards/episodes/steps, build network."""
        super(A2C, self).__init__()

        # saving and summary writing
        if FLAGS.save_dir:
            save_dir = FLAGS.save_dir
        if FLAGS.ckpt_name:
            ckpt_name = FLAGS.ckpt_name
        if FLAGS.summary_path:
            summary_path = FLAGS.summary_path

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # agent hyperparameters
        self.trajectory_training_steps = trajectory_training_steps

        # other parameters
        self.training = training

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.AtariNet(
            screen_dimensions=feature_screen_size,
            minimap_dimensions=feature_minimap_size,
            learning_rate=learning_rate,
            value_gradient_strength=value_gradient_strength,
            regularization_strength=regularization_strength,
            save_path=self.save_path,
            summary_path=summary_path)

        print("Done.")

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
        else:
            self._tf_init_op()

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.reward = 0

        if self.training:
            self.last_action = None
            self.state_buffer = deque(maxlen=self.trajectory_training_steps)
            self.action_buffer = deque(maxlen=self.trajectory_training_steps)
            self.reward_buffer = deque(maxlen=self.trajectory_training_steps)
            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode + 1)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        terminal = False
        if self.training and obs.step_type == 2:
            self._handle_episode_end()
            terminal = True

        # get observations of state
        observation = obs.observation
        # expand so they form a batch of 1
        screen_features = observation.feature_screen
        minimap_features = observation.feature_minimap
        flat_features = observation.player
        available_actions = observation.available_actions

        # sample action (function identifier and arguments) from policy
        action_id, args, arg_types = self._sample_action(
            screen_features,
            minimap_features,
            flat_features,
            available_actions)

        # train model
        if self.training:
            if self.last_action:
                # most recent steps on the left of the deques
                self.state_buffer.appendleft((screen_features,
                                              minimap_features,
                                              flat_features))
                self.action_buffer.appendleft(self.last_action)
                self.reward_buffer.appendleft(obs.reward)

            # cut trajectory and train model
            if self.steps % self.trajectory_training_steps == 0 or terminal:
                self._train_network(terminal)

            self.last_action = [action_id, args, arg_types]

        return FunctionCall(action_id, args)

    def _sample_action(self,
                       screen_features,
                       minimap_features,
                       flat_features,
                       available_actions):
        """Sample actions and arguments from policy output layers."""
        screen_features = np.expand_dims(screen_features, 0)
        minimap_features = np.expand_dims(minimap_features, 0)
        flat_features = np.expand_dims(flat_features, 0)

        action_mask = np.zeros(len(FUNCTIONS), dtype=np.int32)
        action_mask[available_actions] = 1

        feed_dict = {self.network.screen_features: screen_features,
                     self.network.minimap_features: minimap_features,
                     self.network.flat_features: flat_features}

        function_id_policy = self.sess.run(
            self.network.function_policy,
            feed_dict=feed_dict)

        function_id_policy *= action_mask
        function_ids = np.arange(len(function_id_policy))

        # renormalize distribution over function identifiers
        function_id_policy /= np.sum(function_id_policy)

        # sample function identifier
        action_id = np.random.choice(
            function_ids,
            p=np.squeeze(function_id_policy))

        # sample function arguments
        arg_types = FUNCTION_TYPES[FUNCTIONS[action_id].function_type]
        args = []
        for arg_type in arg_types:
            if len(arg_type.sizes) > 1:
                # this is a spatial action
                x_policy = self.sess.run(
                    self.network.argument_policy[str(arg_type) + "x"],
                    feed_dict=feed_dict)

                y_policy = self.sess.run(
                    self.network.argument_policy[str(arg_type) + "y"],
                    feed_dict=feed_dict)

                x_policy = np.squeeze(x_policy)
                x_ids = np.arange(len(x_policy))
                x = np.random.choice(x_ids, p=x_policy)

                y_policy = np.squeeze(y_policy)
                y_ids = np.arange(len(y_policy))
                y = np.random.choice(y_ids, p=y_policy)
                args.append([x, y])
            else:
                arg_policy = self.sess.run(
                    self.network.argument_policy[str(arg_type)],
                    feed_dict=feed_dict)

                arg_policy = np.squeeze(arg_policy)
                arg_ids = np.arange(len(arg_policy))
                arg_index = np.random.choice(arg_ids, p=arg_policy)
                args.append([arg_index])

        return action_id, args, arg_types

    def _handle_episode_end(self):
        """Save weights and write summaries."""
        # save current model
        self.network.save_model(self.sess)
        print("Model Saved")

        # write summaries from last episode
        self.network.write_summary(
            self.sess, self.reward)
        print("Summary Written")

        # increment global training episode
        self.network.increment_global_episode_op(self.sess)

    def _tf_init_op(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _train_network(self, terminal):
        feed_dict = self._get_batch(terminal)
        self.network.optimizer_op(self.sess, feed_dict)

    def _get_batch(self, terminal):
        # state
        screen = [each[0] for each in self.state_buffer]
        minimap = [each[1] for each in self.state_buffer]
        flat = [each[2] for each in self.state_buffer]

        # actions and arguments
        actions = [each[0] for each in self.action_buffer]
        actions = np.eye(len(FUNCTIONS))[actions]  # one-hot encode actions

        args = [each[1] for each in self.action_buffer]
        arg_types = [each[2] for each in self.action_buffer]

        # calculate discounted rewards
        raw_rewards = list(self.reward_buffer)
        if terminal:
            value = 0
        else:
            value = np.squeeze(self.sess.run(
                self.network.value_estimate,
                feed_dict={self.network.screen_features: screen[-1:],
                           self.network.minimap_features: minimap[-1:],
                           self.network.flat_features: flat[-1:]}))

        returns = []
        # n-step discounted rewards from 1 < n < trajectory_training_steps
        for i, reward in enumerate(raw_rewards):
            value = reward + self.discount_factor * value
            returns.append(value)

        feed_dict = {self.network.screen_features: screen,
                     self.network.minimap_features: minimap,
                     self.network.flat_features: flat,
                     self.network.actions: actions,
                     self.network.returns: returns}

        # add args and arg_types to feed_dict
        net_args = self.network.arguments
        batch_size = len(arg_types)

        # first populate feed_dict with zero arrays
        for arg_type in sc2_actions.TYPES:
            if len(arg_type.sizes) > 1:
                if arg_type in SCREEN_TYPES:
                    x_size = feature_screen_size[0]
                    y_size = feature_screen_size[1]
                elif arg_type in MINIMAP_TYPES:
                    x_size = feature_minimap_size[0]
                    y_size = feature_minimap_size[1]

                feed_dict[net_args[str(arg_type) + "x"]] = np.zeros(
                    (batch_size, x_size))
                feed_dict[net_args[str(arg_type) + "y"]] = np.zeros(
                    (batch_size, y_size))

            else:
                feed_dict[net_args[str(arg_type)]] = np.zeros(
                    (batch_size, arg_type.sizes[0]))

        # then one_hot encode args
        for step in range(batch_size):
            for i, arg_type in enumerate(arg_types[step]):
                if len(arg_type.sizes) > 1:
                    arg_key_x = net_args[str(arg_type) + "x"]
                    feed_dict[arg_key_x][step, args[step][i][0]] = 1

                    arg_key_y = net_args[str(arg_type) + "x"]
                    feed_dict[arg_key_y][step, args[step][i][1]] = 1
                else:
                    arg_key = net_args[str(arg_type)]
                    feed_dict[arg_key][step, args[step][i][0]] = 1

        return feed_dict
