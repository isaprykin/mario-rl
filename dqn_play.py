"""Run a trained DQN model to play Mario."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from mario_env import make_env


def configure_tf():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("No GPU detected. Running on CPU.")


def play(args):
    env = make_env(
        env_id=args.env_id,
        action_set=args.action_set,
        frame_size=(args.frame_size, args.frame_size),
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
        seed=args.seed,
    )

    model_dir = Path(args.model_dir)
    if model_dir.is_dir():
        model_path = model_dir / "model.keras"
    else:
        model_path = model_dir
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)

    state, _ = env.reset()
    done = False
    episode_reward = 0.0
    steps = 0

    while not done:
        if args.random_action and np.random.random() < args.random_action:
            action = env.action_space.sample()
        else:
            q_values = model.predict(state[None, ...], verbose=0)[0]
            action = int(np.argmax(q_values))
        state, reward, done, info = env.step(action)
        episode_reward += reward
        steps += 1
        if args.render:
            env.render()

    env.close()
    print(f"Episode reward: {episode_reward:.2f}, steps: {steps}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Play Mario using a trained DQN model.")
    parser.add_argument("--env-id", default="SuperMarioBros-v0")
    parser.add_argument("--action-set", default="complex", choices=["complex", "simple", "right"])
    parser.add_argument("--frame-size", type=int, default=84)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-dir", default="models/mario_dqn")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random-action", type=float, default=0.0)
    return parser


def main():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_tf()
    args = build_arg_parser().parse_args()
    play(args)


if __name__ == "__main__":
    main()
