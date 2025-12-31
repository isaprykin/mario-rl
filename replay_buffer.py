"""Replay buffer with optional prioritized sampling."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, state_shape, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.alpha = float(alpha)
        self.position = 0
        self.full = False

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.uint8)
        self.actions = np.zeros((self.capacity,), dtype=np.int64)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.bool_)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.position

    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.next_states[self.position] = next_state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done

        max_priority = self.priorities.max() if len(self) > 0 else 1.0
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        if self.position == 0:
            self.full = True

    def sample(self, batch_size: int, beta: float = 0.4):
        size = len(self)
        if size == 0:
            raise ValueError("Replay buffer is empty.")

        priorities = self.priorities[:size]
        scaled = priorities ** self.alpha
        prob = scaled / scaled.sum()

        indices = np.random.choice(size, batch_size, p=prob)
        weights = (size * prob[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1.0

        batch = (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
        )
        return batch

    def update_priorities(self, indices, priorities, eps: float = 1e-5):
        priorities = np.asarray(priorities, dtype=np.float32)
        priorities = np.abs(priorities) + eps
        self.priorities[indices] = priorities
