"""Tensorflow graphs that preprocess pysc2 features."""
import tensorflow as tf

from pysc2.lib import static_data

UNIT_TYPES = max(static_data.UNIT_TYPES)
# (channel index, number of categories)
CATEGORICAL_SCREEN_FEATURES = [(1, 4),
                               (2, 2),
                               (3, 2),
                               (4, 17),
                               (5, 5),
                               (6, UNIT_TYPES + 1),
                               (7, 2),
                               (16, 16)]

CATEGORICAL_MINIMAP_FEATURES = [(1, 4),
                                (2, 2),
                                (3, 2),
                                (4, 17),
                                (5, 5),
                                (6, 2)]


def embed_screen_features(screen_features):
    """Embed categorical screen features into continuous space."""
    pass


def embed_minimap_features(minimap_features):
    """Embed categorical minimap features into continuous space."""
