"""Random agents."""

import numpy as np

from absl import flags

from pysc2.agents import base_agent
from pysc2.lib import actions

FLAGS = flags.FLAGS
feature_screen_size = FLAGS.feature_screen_size
feature_minimap_size = FLAGS.feature_minimap_size

FUNCTIONS = actions.FUNCTIONS


class QueueRandomMovements(base_agent.BaseAgent):
    """An agent that queues random screen movements."""

    def step(self, obs):
        """If no units selected, selects army, otherwise moves randomly."""
        super(QueueRandomMovements, self).step(obs)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            x = np.random.randint(0, feature_screen_size[0])
            y = np.random.randint(0, feature_screen_size[1])

            return FUNCTIONS.Move_screen("queued", (x, y))
        else:
            return FUNCTIONS.select_army("select")
