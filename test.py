# This file is for serving the model in Mario.
# Use: $ python -m test

import collections

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

import tensorflow as tf
import numpy as np

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, COMPLEX_MOVEMENT)
state_shape = (84, 84)
        
model = tf.keras.models.load_model('/tmp/models/mario0')

def greyscale(state):
    return tf.image.rgb_to_grayscale([state])[0]

def resize(state):
    return tf.compat.v1.image.resize_images([state], (state_shape[0], state_shape[1]))[0]

def downsample(state):
    state = resize(state)
    state = greyscale(state)
    state = tf.image.per_image_standardization(state)
    return tf.cast(tf.reshape(state, (1,) + state_shape), tf.dtypes.float32)  

def select_action(state):
    if (np.random.random() < 0.0):
        return np.random.choice(len(COMPLEX_MOVEMENT))
    else:
      state = tf.reshape(tf.concat(state, axis=0), (1,4,) + state_shape)
      return np.argmax(model.predict(state))

done = False
episode_count = 0
overlapping_buffer = collections.deque(maxlen=4)
# state = downsample(env.reset())
state = env.reset()
for _ in range(np.random.randint(0, 133)):
  state, _, _, _ = env.step(0)
state = downsample(state)
frame_states = [state, state, state, state]
while not done:
  action = select_action(frame_states)

  total_reward = 0
  for _ in range(4):
      state, reward, done, info = env.step(action)
      if done: break
      total_reward += reward

  overlapping_buffer.append(downsample(state))

  if len(overlapping_buffer) == overlapping_buffer.maxlen:
      frame_states =  [s for s in overlapping_buffer]
      print ('{} -> {} @ {} # {} | {}'.format(
        action, total_reward,
        info['x_pos'], episode_count,
        model.predict(tf.reshape(tf.concat(frame_states, axis=0), (1,4,) + state_shape))))
  env.render()
  episode_count += 1

env.close()      
