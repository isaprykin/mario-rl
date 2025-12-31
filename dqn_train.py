"""Train a Double DQN agent on Super Mario Bros."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from dqn_model import build_q_network
from mario_env import make_env
from replay_buffer import ReplayBuffer


def _env_worker(conn, env_id, action_set, frame_size, frame_skip, frame_stack, seed):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from mario_env import make_env as _make_env

    env = _make_env(
        env_id=env_id,
        action_set=action_set,
        frame_size=frame_size,
        frame_skip=frame_skip,
        frame_stack=frame_stack,
        seed=seed,
    )
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                state, _ = env.reset()
                conn.send(("reset_ok", state))
            elif cmd == "step":
                action = payload
                next_state, reward, done, info = env.step(action)
                conn.send(("step_ok", next_state, reward, done, info))
            elif cmd == "close":
                break
    except EOFError:
        pass
    finally:
        env.close()
        conn.close()


def configure_tf(use_mixed_precision: bool, enable_xla: bool, enable_mem_growth: bool):
    gpus = tf.config.list_physical_devices("GPU")
    if enable_mem_growth and gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass

    if enable_xla:
        tf.config.optimizer.set_jit(True)

    if use_mixed_precision:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")

    if gpus:
        print(f"Using GPU(s): {[gpu.name for gpu in gpus]}")
    else:
        print("No GPU detected. Training will run on CPU.")


def _epsilon_by_step(step: int, eps_start: float, eps_final: float, eps_decay: int):
    if eps_decay <= 0:
        return eps_final
    return eps_final + (eps_start - eps_final) * np.exp(-1.0 * step / eps_decay)


def _beta_by_step(step: int, beta_start: float, beta_frames: int):
    if beta_frames <= 0:
        return 1.0
    return min(1.0, beta_start + step * (1.0 - beta_start) / beta_frames)


def _load_state(state_path: Path):
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_path: Path, payload: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def train(args):
    use_parallel = args.parallel_envs > 0
    num_envs = args.parallel_envs if use_parallel else args.num_envs
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    envs = []
    conns = []
    workers = []
    if use_parallel:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        for idx in range(num_envs):
            seed = args.seed + idx if args.seed is not None else None
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_env_worker,
                args=(
                    child_conn,
                    args.env_id,
                    args.action_set,
                    (args.frame_size, args.frame_size),
                    args.frame_skip,
                    args.frame_stack,
                    seed,
                ),
            )
            proc.daemon = True
            proc.start()
            child_conn.close()
            conns.append(parent_conn)
            workers.append(proc)
    else:
        for idx in range(num_envs):
            seed = args.seed + idx if args.seed is not None else None
            envs.append(
                make_env(
                    env_id=args.env_id,
                    action_set=args.action_set,
                    frame_size=(args.frame_size, args.frame_size),
                    frame_skip=args.frame_skip,
                    frame_stack=args.frame_stack,
                    seed=seed,
                )
            )

    if use_parallel:
        temp_env = make_env(
            env_id=args.env_id,
            action_set=args.action_set,
            frame_size=(args.frame_size, args.frame_size),
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
        )
        action_count = temp_env.action_space.n
        temp_env.close()
    else:
        action_count = envs[0].action_space.n
    input_shape = (args.frame_stack, args.frame_size, args.frame_size)

    model = build_q_network(action_count, input_shape, learning_rate=args.learning_rate)
    target_model = build_q_network(action_count, input_shape, learning_rate=args.learning_rate)
    target_model.set_weights(model.get_weights())

    replay = ReplayBuffer(args.replay_size, input_shape, alpha=args.prioritized_alpha)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"
    state_path = model_dir / "training_state.json"

    if args.resume:
        state = _load_state(state_path) or {}
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            if model.optimizer is None:
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                    loss=tf.keras.losses.Huber(),
                )
            target_model = build_q_network(action_count, input_shape, learning_rate=args.learning_rate)
            target_model.set_weights(model.get_weights())
            start_step = int(state.get("step", 0))
            start_episode = int(state.get("episode", 0))
            print(f"Resuming from step {start_step}, episode {start_episode}")
        else:
            start_step = 0
            start_episode = 0
    else:
        start_step = 0
        start_episode = 0

    writer = tf.summary.create_file_writer(str(model_dir / "logs"))

    global_step = start_step
    episode = start_episode
    best_reward = -float("inf")
    epsilon = _epsilon_by_step(global_step, args.eps_start, args.eps_final, args.eps_decay)
    last_log_time = time.time()
    last_log_step = global_step

    def train_on_replay(step):
        if step < args.prefetch_steps:
            return
        if len(replay) >= args.batch_size and step % args.train_freq == 0:
            beta = _beta_by_step(step, args.beta_start, args.beta_frames)
            (
                batch_states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ) = replay.sample(args.batch_size, beta=beta)

            q_values = model.predict(batch_states, verbose=0)
            next_q_online = model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q_online, axis=1)
            next_q_target = target_model.predict(next_states, verbose=0)
            next_q = next_q_target[np.arange(args.batch_size), next_actions]

            targets = rewards + (1.0 - dones.astype(np.float32)) * args.gamma * next_q
            td_errors = targets - q_values[np.arange(args.batch_size), actions]
            q_values[np.arange(args.batch_size), actions] = targets

            model.train_on_batch(batch_states, q_values, sample_weight=weights)
            replay.update_priorities(indices, td_errors)

    def log_step(step, eps, mean_reward, mean_steps):
        nonlocal last_log_time, last_log_step
        with writer.as_default():
            tf.summary.scalar("epsilon", eps, step=step)
            tf.summary.scalar("episode_reward_mean", mean_reward, step=step)
            tf.summary.scalar("replay_size", len(replay), step=step)
            tf.summary.scalar("episode_steps_mean", mean_steps, step=step)
        now = time.time()
        step_delta = max(1, step - last_log_step)
        time_delta = max(1e-6, now - last_log_time)
        sps = step_delta / time_delta
        print(
            f"[step {step}] ep {episode} mean_ep_steps {mean_steps:.1f} "
            f"mean_ep_reward {mean_reward:.2f} epsilon {eps:.3f} "
            f"replay {len(replay)} sps {sps:.1f}",
            flush=True,
        )
        last_log_time = now
        last_log_step = step

    def save_checkpoint(step, eps):
        model.save(model_path, overwrite=True, include_optimizer=False)
        _save_state(
            state_path,
            {
                "step": step,
                "episode": episode,
                "epsilon": float(eps),
                "timestamp": time.time(),
            },
        )

    def finalize_episode(ep_return, ep_steps):
        nonlocal best_reward, episode
        if ep_return > best_reward:
            best_reward = ep_return
            if args.save_best:
                best_path = model_dir / "best.keras"
                model.save(best_path, overwrite=True, include_optimizer=False)

        with writer.as_default():
            tf.summary.scalar("episode_return", ep_return, step=global_step)
            tf.summary.scalar("episode_steps", ep_steps, step=global_step)

        print(
            f"[episode {episode}] return {ep_return:.2f} "
            f"steps {ep_steps} total_step {global_step}",
            flush=True,
        )
        episode += 1

    states: List[np.ndarray] = []
    episode_rewards: List[float] = []
    episode_steps: List[int] = []

    if use_parallel:
        for conn in conns:
            conn.send(("reset", None))
        for conn in conns:
            msg, state = conn.recv()
            if msg != "reset_ok":
                raise RuntimeError("Worker failed to reset.")
            states.append(state)
            episode_rewards.append(0.0)
            episode_steps.append(0)

        while global_step < args.total_steps:
            epsilon = _epsilon_by_step(global_step, args.eps_start, args.eps_final, args.eps_decay)
            state_batch = np.stack(states, axis=0)
            q_batch = model.predict(state_batch, verbose=0)
            actions = []
            for i in range(num_envs):
                if np.random.random() < epsilon:
                    actions.append(np.random.randint(action_count))
                else:
                    actions.append(int(np.argmax(q_batch[i])))

            for conn, action in zip(conns, actions):
                conn.send(("step", action))

            for env_index, conn in enumerate(conns):
                msg, next_state, reward, done, info = conn.recv()
                if msg != "step_ok":
                    raise RuntimeError("Worker failed to step.")

                reward = np.clip(reward, -1.0, 1.0)
                episode_rewards[env_index] += reward
                episode_steps[env_index] += 1
                replay.add(states[env_index], actions[env_index], reward, next_state, done)
                states[env_index] = next_state

                train_on_replay(global_step)
                if global_step % args.target_update == 0 and global_step > 0:
                    target_model.set_weights(model.get_weights())

                if global_step % args.log_interval == 0:
                    mean_reward = float(np.mean(episode_rewards))
                    mean_steps = float(np.mean(episode_steps))
                    log_step(global_step, epsilon, mean_reward, mean_steps)

                if global_step % args.save_interval == 0 and global_step > 0:
                    save_checkpoint(global_step, epsilon)

                global_step += 1

                if done:
                    finalize_episode(episode_rewards[env_index], episode_steps[env_index])
                    conn.send(("reset", None))
                    msg, reset_state = conn.recv()
                    if msg != "reset_ok":
                        raise RuntimeError("Worker failed to reset.")
                    states[env_index] = reset_state
                    episode_rewards[env_index] = 0.0
                    episode_steps[env_index] = 0

                if global_step >= args.total_steps:
                    break
    else:
        for env in envs:
            state, _ = env.reset()
            states.append(state)
            episode_rewards.append(0.0)
            episode_steps.append(0)

        while global_step < args.total_steps:
            for env_index, env in enumerate(envs):
                if global_step >= args.total_steps:
                    break

                epsilon = _epsilon_by_step(global_step, args.eps_start, args.eps_final, args.eps_decay)
                if np.random.random() < epsilon:
                    action = np.random.randint(action_count)
                else:
                    q_values = model.predict(states[env_index][None, ...], verbose=0)[0]
                    action = int(np.argmax(q_values))

                next_state, reward, done, info = env.step(action)
                reward = np.clip(reward, -1.0, 1.0)
                episode_rewards[env_index] += reward
                episode_steps[env_index] += 1
                replay.add(states[env_index], action, reward, next_state, done)
                states[env_index] = next_state

                train_on_replay(global_step)
                if global_step % args.target_update == 0 and global_step > 0:
                    target_model.set_weights(model.get_weights())

                if global_step % args.log_interval == 0:
                    mean_reward = float(np.mean(episode_rewards))
                    mean_steps = float(np.mean(episode_steps))
                    log_step(global_step, epsilon, mean_reward, mean_steps)

                if global_step % args.save_interval == 0 and global_step > 0:
                    save_checkpoint(global_step, epsilon)

                global_step += 1

                if done:
                    finalize_episode(episode_rewards[env_index], episode_steps[env_index])
                    reset_state, _ = env.reset()
                    states[env_index] = reset_state
                    episode_rewards[env_index] = 0.0
                    episode_steps[env_index] = 0

    model.save(model_path, overwrite=True, include_optimizer=False)
    _save_state(
        state_path,
        {"step": global_step, "episode": episode, "epsilon": float(epsilon), "timestamp": time.time()},
    )
    if use_parallel:
        for conn in conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        for proc in workers:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.0)
    else:
        for env in envs:
            env.close()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a Double DQN agent for Mario.")
    parser.add_argument("--env-id", default="SuperMarioBros-v0")
    parser.add_argument("--action-set", default="complex", choices=["complex", "simple", "right"])
    parser.add_argument("--frame-size", type=int, default=84)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--parallel-envs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replay-size", type=int, default=20000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--prefetch-steps", type=int, default=0)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-final", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=200000)

    parser.add_argument("--prioritized-alpha", type=float, default=0.6)
    parser.add_argument("--beta-start", type=float, default=0.4)
    parser.add_argument("--beta-frames", type=int, default=200000)

    parser.add_argument("--target-update", type=int, default=10000)
    parser.add_argument("--total-steps", type=int, default=500000)
    parser.add_argument("--model-dir", default="models/mario_dqn")
    parser.add_argument("--save-interval", type=int, default=50000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--no-mem-growth", action="store_true")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_tf(
        use_mixed_precision=args.mixed_precision,
        enable_xla=args.xla,
        enable_mem_growth=not args.no_mem_growth,
    )
    train(args)


if __name__ == "__main__":
    main()
