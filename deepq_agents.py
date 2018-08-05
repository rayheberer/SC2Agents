"""Deep Q-learning agents."""
import numpy as np
import os
import tensorflow as tf

# local submodule
import networks.value_estimators as nets

from absl import flags

from collections import deque

from pysc2.agents import base_agent
from pysc2.lib import actions as sc2_actions

FLAGS = flags.FLAGS

# agent interface settings (defaults specified in pysc2.bin.agent)
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

# neural network hyperparameters
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate.")
flags.DEFINE_float("discount_factor", 0.99, "Discount factor.")

# agent hyperparameters
flags.DEFINE_float("epsilon_max", 1, "Initial exploration probability.")
flags.DEFINE_float("epsilon_min", 0.02, "Final exploration probability.")
flags.DEFINE_integer("epsilon_decay_steps", 100000, "Steps for linear decay.")
flags.DEFINE_integer("train_every", 1, "Steps between training batches.")

flags.DEFINE_integer("max_memory", 4096, "Experience Replay buffer size.")
flags.DEFINE_integer("batch_size", 32, "Training batch size.")

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
        self.save_path = FLAGS.save_dir + "DQNPlayerRelativeMoveScreen.ckpt"
        print("Building model...")
        self.network = nets.PlayerRelativeMovementCNN(
            spacial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=FLAGS.summary_path)
        print("Done.")

        # initialize Experience Replay memory buffer
        self.memory = Memory(FLAGS.max_memory)
        self.batch_size = FLAGS.batch_size

        self.last_state = None
        self.last_action = None

        # initialize session
        self.sess = tf.Session()
        if os.path.isfile(self.save_path + ".index"):
            self.network.load(self.sess)
        else:
            self.network.run_init_op(self.sess)

    def reset(self):
        """Handle the beginning of new episodes."""
        self.episodes += 1
        self.network.increment_global_episode_op(self.sess)
        score = self.reward
        self.reward = 0

        self.last_state = None
        self.last_action = None

        # don't do anything else for 1st episode
        if self.episodes > 1:

            # save current model
            self.network.save_model(self.sess)
            print("Model Saved")

            # write summaries from last episode
            states, actions, targets = self._get_batch()
            self.network.write_summary(
                self.sess, states, actions, targets, score)
            print("Summary Written")

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        self.steps += 1
        self.reward += obs.reward

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            state = obs.observation.feature_screen.player_relative

            # predict an action to take and take it
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
        step = self.network.global_step.eval(session=self.sess)
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
                self.network.flatten,
                feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y

    def _train_network(self):
        states, actions, targets = self._get_batch()
        self.network.optimizer_op(self.sess, states, actions, targets)

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
            self.network.output,
            feed_dict={self.network.inputs: next_states})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets
