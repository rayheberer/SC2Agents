"""Deep Q-learning agents."""
import numpy as np

from absl import flags

from pysc2.agents import base_agent
from pysc2.lib import actions

FLAGS = flags.FLAGS
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

FUNCTIONS = actions.FUNCTIONS


class DQNPlayerRelativeMoveScreen(base_agent.BaseAgent):
    """A DQN that receives `player_relative` features and takes movements."""

    def step(self, obs):
        """If no units selected, selects army, otherwise move."""
        super(DQNPlayerRelativeMoveScreen, self).step(obs)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            raise NotImplementedError
        else:
            return FUNCTIONS.select_army("select")
