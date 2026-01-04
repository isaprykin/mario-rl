"""Environment helpers and preprocessing for Super Mario Bros."""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Deque, Tuple

import numpy as np
import tensorflow as tf

import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import actions as mario_actions


def _reset_env(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        obs, info = out, {}
    return obs, info


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    return obs, reward, bool(done), info


def _preprocess_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Convert RGB frame to grayscale resized uint8 array (H, W)."""
    frame = tf.image.rgb_to_grayscale(frame)
    frame = tf.image.resize(frame, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(tf.squeeze(frame, axis=-1), tf.uint8).numpy()


@dataclass
class MarioEnv:
    env_id: str = "SuperMarioBros-v0"
    action_set: str = "complex"
    frame_size: Tuple[int, int] = (84, 84)
    frame_skip: int = 4
    frame_stack: int = 4
    seed: int | None = None

    def __post_init__(self):
        base_env = gym_super_mario_bros.make(self.env_id)
        action_space = _action_set(self.action_set)
        self.env = JoypadSpace(base_env, action_space)
        if self.seed is not None:
            try:
                self.env.reset(seed=self.seed)
            except TypeError:
                self.env.seed(self.seed)
        self.action_space = self.env.action_space
        self._frames: Deque[np.ndarray] = collections.deque(maxlen=self.frame_stack)

    def reset(self):
        obs, info = _reset_env(self.env)
        frame = _preprocess_frame(obs, self.frame_size)
        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(frame)
        return self._get_state(), info

    def step(self, action: int):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self.frame_skip):
            obs, reward, done, info = _step_env(self.env, action)
            total_reward += reward
            if done:
                break
        if obs is None:
            raise RuntimeError("Environment step returned no observation.")
        frame = _preprocess_frame(obs, self.frame_size)
        self._frames.append(frame)
        return self._get_state(), float(total_reward), done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()

    def _get_state(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)


def _action_set(name: str):
    name = name.lower()
    if name in {"complex", "complex_movement"}:
        return mario_actions.COMPLEX_MOVEMENT
    if name in {"simple", "simple_movement"}:
        return mario_actions.SIMPLE_MOVEMENT
    if name in {"right", "right_only"}:
        return mario_actions.RIGHT_ONLY
    raise ValueError(f"Unknown action_set: {name}")


def make_env(
    env_id: str = "SuperMarioBros-v0",
    action_set: str = "complex",
    frame_size: Tuple[int, int] = (84, 84),
    frame_skip: int = 4,
    frame_stack: int = 4,
    seed: int | None = None,
):
    return MarioEnv(
        env_id=env_id,
        action_set=action_set,
        frame_size=frame_size,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        seed=seed,
    )
