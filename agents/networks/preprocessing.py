"""Tensorflow graphs that preprocess pysc2 features."""
import tensorflow as tf

from pysc2.lib import features as sc2_features

#  get (channel index, categories) for categorical screen/minimap features
SCREEN_FEATURES = [
    (feature.type, feature.scale)
    for feature in sc2_features.SCREEN_FEATURES
]

MINIMAP_FEATURES = [
    (feature.type, feature.scale)
    for feature in sc2_features.MINIMAP_FEATURES
]


def preprocess_spatial_features(features, screen=True):
    """Embed categorical spatial features, log transform numeric features."""
    if screen:
        feature_specs = SCREEN_FEATURES
    else:
        feature_specs = MINIMAP_FEATURES

    # transpose from (channels, y, x) to (x, y, channels)
    transposed = tf.transpose(
        features,
        perm=[2, 1, 0],
        name="transpose")

    # expand dims so conv layers will work
    expanded = tf.expand_dims(
        transposed,
        axis=0,
        name="expand_dims")

    preprocess_ops = []
    for index, (feature_type, scale) in enumerate(feature_specs):
        layer = expanded[:, :, :, index]

        if feature_type == sc2_features.FeatureType.CATEGORICAL:
            # one-hot encode in channel dimension -> 1x1 convolution
            one_hot = tf.one_hot(
                layer,
                depth=scale,
                axis=-1,
                name="one_hot")

            embed = tf.layers.conv2d(
                inputs=one_hot,
                filters=1,
                kernel_size=[1, 1],
                strides=[1, 1],
                padding="SAME")

            preprocess_ops.append(tf.squeeze(embed))
        else:
            transform = tf.log(
                tf.cast(layer, tf.float32) + 1.,
                name="log")

            preprocess_ops.append(tf.squeeze(transform))

    preprocessed = tf.stack(preprocess_ops, -1)
    return preprocessed
