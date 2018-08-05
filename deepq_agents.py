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

    def __init__(self,
                 learning_rate=1e-6,
                 discount_factor=0.99,
                 epsilon_max=1.0,
                 epsilon_min=0.01,
                 epsilon_decay_steps=100000,
                 train_frequency=1,
                 target_update_frequency=40,
                 save_dir="./checkpoints/",
                 ckpt_name="DQNMoveOnly",
                 summary_path="./tensorboard/deepq",
                 max_memory=4096,
                 batch_size=32,
                 indicate_nonrandom_action=True):
        """Initialize rewards/episodes/steps, build network."""
        super(DQNMoveOnly, self).__init__()

        # neural net hyperparameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # agent hyperparameters
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency

        # other parameters
        self.indicate_nonrandom_action = indicate_nonrandom_action

        # build network
        self.save_path = save_dir + ckpt_name + ".ckpt"
        print("Building models...")
        self.network = nets.PlayerRelativeMovementCNN(
            spacial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            save_path=self.save_path,
            summary_path=summary_path)

        self.target_net = nets.PlayerRelativeMovementCNN(
            spacial_dimensions=feature_screen_size,
            learning_rate=self.learning_rate,
            name='DQNTarget')
        print("Done.")

        # initialize Experience Replay memory buffer
        self.memory = Memory(max_memory)
        self.batch_size = batch_size

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

        global_episode = self.network.global_episode.eval(session=self.sess)
        print("Global episode:", global_episode)

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
            x, y, action_type = self._epsilon_greedy_action_selection(state)

            # update online DQN
            if (self.steps % self.train_frequency == 0 and
                    len(self.memory) > self.batch_size):
                self._train_network()

            # update network used to estimate TD targets
            if self.steps % self.target_update_frequency == 0:
                self._update_target_network()

            # add experience to memory
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

            if self.indicate_nonrandom_action and action_type == 'nonrandom':
                # cosmetic difference between random and Q based actions
                return FUNCTIONS.Attack_screen("now", (x, y))
            else:
                return FUNCTIONS.Move_screen("now", (x, y))
        else:
            return FUNCTIONS.select_army("select")

    def _update_target_network(self):
        online_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'DQN')
        target_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'DQNTarget')

        update_op = []
        for online_var, target_var in zip(online_vars, target_vars):
            update_op.append(target_var.assign(online_var))

        self.sess.run(update_op)

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

            return x, y, 'random'

        else:
            inputs = np.expand_dims(state, 0)

            q_values = self.sess.run(
                self.network.flatten,
                feed_dict={self.network.inputs: inputs})

            max_index = np.argmax(q_values)
            x, y = np.unravel_index(max_index, feature_screen_size)
            return x, y, 'nonrandom'

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
            self.target_net.output,
            feed_dict={self.target_net.inputs: next_states})

        targets = [rewards[i] + self.discount_factor * np.max(next_outputs[i])
                   for i in range(self.batch_size)]

        return states, actions, targets
