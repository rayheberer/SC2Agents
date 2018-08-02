import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

FUNCTIONS = actions.FUNCTIONS

class DeepQMineralShards(base_agent.BaseAgent):
	"""A deep Q-learning agent that can only select units and move"""

	def step(self, obs):
		super(DeepQMineralShards, self).step(obs)

		if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
			print(obs.observation.feature_screen.player_relative)
			return FUNCTIONS.no_op()
		else:
			return FUNCTIONS.select_army("select")