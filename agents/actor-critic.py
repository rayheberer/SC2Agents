"""Actor-critic agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import agents.networks.policy_value_estimators as nets

from absl import flags

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


class A2C(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""

    def __init__(self,
                 learning_rate=FLAGS.learning_rate,
                 discount_factor=FLAGS.discount_factor,
                 training=FLAGS.training,
                 save_dir="./checkpoints",
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

        # other parameters
        self.training = training

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        tf.reset_default_graph()
        self.network = nets.AtariNet(
            screen_dimensions=feature_screen_size,
            minimap_dimensions=feature_minimap_size,
            learning_rate=self.learning_rate,
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
            episode = self.network.global_episode.eval(session=self.sess)
            print("Global training episode:", episode)

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        # handle end of episode if terminal step
        if self.training and obs.step_type == 2:
            self._handle_episode_end()

        # get observations of state
        observation = obs.observation
        screen_features = observation.feature_screen
        minimap_features = observation.feature_minimap
        flat_features = observation.player
        available_actions = observation.available_actions

        # sample action (function identifier and arguments) from policy
        action_id, args = self._sample_action(
            screen_features,
            minimap_features,
            flat_features,
            available_actions)

        return FunctionCall(action_id, args)

    def _sample_action(self,
                       screen_features,
                       minimap_features,
                       flat_features,
                       available_actions):
        """Sample actions and arguments from policy output layers."""
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
                    self.network.argument_policies[str(arg_type) + "x"],
                    feed_dict=feed_dict)

                y_policy = self.sess.run(
                    self.network.argument_policies[str(arg_type) + "y"],
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
                    self.network.argument_policies[str(arg_type)],
                    feed_dict=feed_dict)

                arg_policy = np.squeeze(arg_policy)
                arg_ids = np.arange(len(arg_policy))
                arg_index = np.random.choice(arg_ids, p=arg_policy)
                args.append([arg_index])

        return action_id, args

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
