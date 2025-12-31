"""Q-network model for Mario DQN."""

from __future__ import annotations

import tensorflow as tf


def build_q_network(
    action_count: int,
    input_shape=(4, 84, 84),
    learning_rate: float = 1e-4,
):
    inputs = tf.keras.Input(shape=input_shape, dtype=tf.uint8, name="frames")
    # Normalize to [0,1] and convert to channels-last for Conv2D.
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Permute((2, 3, 1))(x)
    x = tf.keras.layers.Conv2D(
        32, (8, 8), strides=(4, 4), activation="relu", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, (4, 4), strides=(2, 2), activation="relu", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), strides=(1, 1), activation="relu", kernel_initializer="he_normal"
    )(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(x)
    outputs = tf.keras.layers.Dense(action_count, name="q_values")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mario_dqn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
    )
    return model
