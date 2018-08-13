"""Actor-critic agents."""
import numpy as np
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


class A2C(base_agent.BaseAgent):
    """Synchronous version of DeepMind baseline Advantage actor-critic."""
    pass
