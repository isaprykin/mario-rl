# This file is for serving the model in Mario.
# Use: $ python -m test 

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

import tensorflow as tf
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, RIGHT_ONLY)

model = tf.keras.models.load_model('/tmp/mario0')
state_shape = (84, 84, 1)

def greyscale(state):
    return tf.image.rgb_to_grayscale([state])[0]

def resize(state):
    return tf.compat.v1.image.resize_images([state], (state_shape[0], state_shape[1]))[0]

def downsample(state):
    state = resize(state)
    state = greyscale(state)
    # state = (state - 128) / 128
    state = (state - 155) / 75
    return tf.cast(tf.reshape(state, (1,) + state_shape), tf.dtypes.bfloat16)  

def policy(state):
  return np.argmax(model.predict(state))

state = env.reset()
done = False
episode_count = 0
while not done:
    state = downsample(state)
    action = policy(state)
    state, reward, done, info = env.step(action)
    # print ('{} -> {} @ {} # {} | {}'.format(
    #   action, reward,
    #   info['x_pos'], episode_count,
    #   model.predict(downsample(state))))
    if done:
      break
    env.render()
    episode_count += 1

env.close()



