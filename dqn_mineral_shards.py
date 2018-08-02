import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

FUNCTIONS = actions.FUNCTIONS

# these are defaults set in pysc2.bin.agent
feature_screen_size = 84
feature_minimap_size = 64


class DeepQMineralShards(base_agent.BaseAgent):
    """A deep Q-learning agent that can only select units and move"""

    def step(self, obs):
        super(DeepQMineralShards, self).step(obs)

        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            move_to = np.random.randint(0, feature_screen_size, 2)
            return FUNCTIONS.Move_screen("queued", move_to)
        else:
            return FUNCTIONS.select_army("select")