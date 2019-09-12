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
state_shape = (56, 56, 1)

def greyscale(state):
    return tf.image.rgb_to_grayscale([state])[0]

def resize(state):
    return tf.compat.v1.image.resize_images([state], (state_shape[0], state_shape[1]))[0]

def downsample(state):
    state = resize(state)
    state = greyscale(state)
    state = (state - 128) / 128    
    return tf.cast(tf.reshape(state, (1,) + state_shape), tf.dtypes.bfloat16)  

def policy(state):
  return np.argmax(model.predict(state))

state = env.reset()
for step in range(600):
    state = downsample(state)
    action = policy(state)
    state, reward, done, info = env.step(action)
    print ('{} - {}'.format(reward, model.predict(downsample(state))))
    if done:
      break
    env.render()

env.close()
