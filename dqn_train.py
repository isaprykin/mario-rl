"""Train a Double DQN agent on Super Mario Bros."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from dqn_model import build_q_network
from mario_env import make_env
from replay_buffer import ReplayBuffer

# IPC message constants
MSG_READY = "ready"
MSG_WEIGHTS_READY = "weights_ready"
MSG_CLOSE = "close"

# Training constants
ROLLING_WINDOW_SIZE = 100
PROCESS_JOIN_TIMEOUT = 2.0
DEFAULT_QUEUE_SIZE = 2000


class TimingStats:
    def __init__(self):
        self._totals = {}
        self._counts = {}
        self._order = []

    def add(self, name: str, duration: float):
        if name not in self._totals:
            self._totals[name] = 0.0
            self._counts[name] = 0
            self._order.append(name)
        self._totals[name] += duration
        self._counts[name] += 1

    def summary_and_reset(self) -> str:
        parts = []
        for name in self._order:
            count = self._counts.get(name, 0)
            total = self._totals.get(name, 0.0)
            avg_ms = (total / count * 1000.0) if count else 0.0
            parts.append(f"{name} {avg_ms:.2f}ms x{count}")
        self._totals.clear()
        self._counts.clear()
        self._order.clear()
        return " | ".join(parts)


class SharedWeights:
    """Shared memory buffer for model weights with version tracking."""

    def __init__(self, weight_shapes, shm_name=None, version_counter=None):
        from multiprocessing import shared_memory

        self._shapes = weight_shapes
        self._offsets = []
        self._sizes = []
        total_floats = 0
        for shape in weight_shapes:
            self._offsets.append(total_floats)
            size = int(np.prod(shape))
            self._sizes.append(size)
            total_floats += size

        self._total_bytes = total_floats * 4  # float32
        self._version = version_counter

        if shm_name is None:
            self._shm = shared_memory.SharedMemory(create=True, size=self._total_bytes)
            self._owns_shm = True
        else:
            self._shm = shared_memory.SharedMemory(name=shm_name)
            self._owns_shm = False

        self._buffer = np.ndarray((total_floats,), dtype=np.float32, buffer=self._shm.buf)
        self._local_version = -1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def name(self):
        return self._shm.name

    def _read_weights(self):
        """Reconstruct weight arrays from shared buffer."""
        return [
            self._buffer[offset : offset + size].reshape(shape).copy()
            for shape, offset, size in zip(self._shapes, self._offsets, self._sizes)
        ]

    def write(self, weights):
        """Write weights to shared memory and increment version."""
        for w, offset, size in zip(weights, self._offsets, self._sizes):
            self._buffer[offset : offset + size] = w.ravel()
        if self._version is not None:
            self._version.value += 1

    def read_if_new(self):
        """Read weights if version changed, return None otherwise."""
        if self._version is None:
            return None
        current_version = self._version.value
        if current_version == self._local_version:
            return None
        self._local_version = current_version
        return self._read_weights()

    def read(self):
        """Force read weights (for initial load)."""
        if self._version is not None:
            self._local_version = self._version.value
        return self._read_weights()

    def close(self):
        self._shm.close()
        if self._owns_shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass


@dataclass
class ActorConfig:
    """Configuration for actor workers (picklable across processes)."""
    env_id: str
    action_set: str
    frame_size: tuple
    frame_skip: int
    frame_stack: int
    seed: int | None
    action_count: int
    input_shape: tuple
    eps_start: float
    eps_final: float
    eps_decay: int
    weight_shapes: list
    shm_weights_name: str


def _actor_worker(conn, transition_queue, stats_queue, config: ActorConfig, shared_step, weight_version, dropped_transitions):
    """Actor process that collects transitions using its own environment and model."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from mario_env import make_env as _make_env

    if config.seed is not None:
        np.random.seed(config.seed)

    env = _make_env(
        env_id=config.env_id,
        action_set=config.action_set,
        frame_size=config.frame_size,
        frame_skip=config.frame_skip,
        frame_stack=config.frame_stack,
        seed=config.seed,
    )
    model = build_q_network(config.action_count, config.input_shape)
    shared_weights = SharedWeights(config.weight_shapes, config.shm_weights_name, weight_version)

    try:
        conn.send(MSG_READY)
        msg = conn.recv()
        if msg != MSG_WEIGHTS_READY:
            raise RuntimeError("Actor did not receive weights_ready signal.")
        model.set_weights(shared_weights.read())

        state, _ = env.reset()
        episode_return = 0.0
        episode_steps = 0

        while True:
            if conn.poll():
                msg = conn.recv()
                if msg == MSG_CLOSE:
                    break

            new_weights = shared_weights.read_if_new()
            if new_weights is not None:
                model.set_weights(new_weights)

            step = int(shared_step.value)
            eps = _epsilon_by_step(step, config.eps_start, config.eps_final, config.eps_decay)

            if np.random.rand() < eps:
                action = np.random.randint(config.action_count)
            else:
                q_values = model(np.expand_dims(state, axis=0), training=False)
                action = int(tf.argmax(q_values[0]).numpy())

            next_state, reward, done, _ = env.step(action)
            try:
                transition_queue.put_nowait((state, action, reward, next_state, done))
            except queue.Full:
                with dropped_transitions.get_lock():
                    dropped_transitions.value += 1

            episode_return += reward
            episode_steps += 1

            if done:
                stats_queue.put((episode_return, episode_steps))
                state, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0
            else:
                state = next_state
    except EOFError:
        pass
    finally:
        env.close()
        conn.close()
        shared_weights.close()


def configure_tf(use_mixed_precision: bool, enable_xla: bool, enable_mem_growth: bool) -> None:
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


def _build_train_step(model, target_model, gamma: float, clip_norm: float | None = None):
    huber = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
    gamma_tf = tf.constant(gamma, dtype=tf.float32)

    @tf.function(reduce_retracing=True)
    def train_step(states, actions, rewards, next_states, dones, weights):
        actions = tf.cast(actions, tf.int32)
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(dones, tf.float32)
        weights = tf.cast(weights, tf.float32)

        with tf.GradientTape() as tape:
            batch_size = tf.shape(states)[0]
            combined_states = tf.concat([states, next_states], axis=0)
            combined_q = model(combined_states, training=True)
            q_values = combined_q[:batch_size]
            next_q_online = combined_q[batch_size:]

            q_taken = tf.gather(q_values, actions, axis=1, batch_dims=1)
            next_actions = tf.argmax(next_q_online, axis=1, output_type=tf.int32)

            next_q_target = target_model(next_states, training=False)
            next_q = tf.gather(next_q_target, next_actions, axis=1, batch_dims=1)
            next_q = tf.cast(next_q, tf.float32)  # For mixed precision compatibility
            q_taken = tf.cast(q_taken, tf.float32)

            targets = rewards + (1.0 - dones) * gamma_tf * next_q
            td_errors = targets - q_taken
            loss = huber(targets, q_taken)
            loss = tf.reduce_mean(loss * weights)

        grads = tape.gradient(loss, model.trainable_variables)
        if clip_norm is not None and clip_norm > 0:
            grads, grad_norm = tf.clip_by_global_norm(grads, clip_norm)
        else:
            grad_norm = tf.linalg.global_norm(grads)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Calculate Q-value stats for logging
        mean_q = tf.reduce_mean(q_taken)
        max_q = tf.reduce_max(q_taken)
        
        return td_errors, loss, mean_q, max_q, grad_norm

    return train_step


def _epsilon_by_step(step: int, eps_start: float, eps_final: float, eps_decay: int) -> float:
    if eps_decay <= 0:
        return eps_final
    return eps_final + (eps_start - eps_final) * np.exp(-1.0 * step / eps_decay)


def _beta_by_step(step: int, beta_start: float, beta_frames: int) -> float:
    if beta_frames <= 0:
        return 1.0
    return min(1.0, beta_start + step * (1.0 - beta_start) / beta_frames)


def _load_state(state_path: Path) -> dict | None:
    if not state_path.exists():
        return None
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(state_path: Path, payload: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def train(args: argparse.Namespace) -> None:
    if args.num_actors < 1:
        raise ValueError("--num-actors must be >= 1")

    conns = []
    workers = []
    input_shape = (args.frame_stack, args.frame_size, args.frame_size)
    temp_env = make_env(
        env_id=args.env_id,
        action_set=args.action_set,
        frame_size=(args.frame_size, args.frame_size),
        frame_skip=args.frame_skip,
        frame_stack=args.frame_stack,
    )
    action_count = temp_env.action_space.n
    temp_env.close()

    ctx = mp.get_context("spawn")
    transition_queue = ctx.Queue(maxsize=args.actor_queue_size)
    stats_queue = ctx.Queue()
    shared_step = ctx.Value("i", 0, lock=False)
    weight_version = ctx.Value("i", 0, lock=False)
    dropped_transitions = ctx.Value("i", 0, lock=True)

    # Build a temporary model to get weight shapes
    temp_model = build_q_network(action_count, input_shape)
    weight_shapes = [w.shape for w in temp_model.get_weights()]
    del temp_model

    # Create shared memory for weights
    shared_weights = SharedWeights(weight_shapes, version_counter=weight_version)

    for idx in range(args.num_actors):
        config = ActorConfig(
            env_id=args.env_id,
            action_set=args.action_set,
            frame_size=(args.frame_size, args.frame_size),
            frame_skip=args.frame_skip,
            frame_stack=args.frame_stack,
            seed=args.seed + idx if args.seed is not None else None,
            action_count=action_count,
            input_shape=input_shape,
            eps_start=args.eps_start,
            eps_final=args.eps_final,
            eps_decay=args.eps_decay,
            weight_shapes=weight_shapes,
            shm_weights_name=shared_weights.name,
        )
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_actor_worker, args=(
            child_conn, transition_queue, stats_queue, config, shared_step, weight_version, dropped_transitions
        ))
        proc.daemon = True
        proc.start()
        child_conn.close()
        msg = parent_conn.recv()
        if msg != MSG_READY:
            raise RuntimeError("Actor failed to report ready state.")
        conns.append(parent_conn)
        workers.append(proc)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"
    state_path = model_dir / "training_state.json"

    start_step, start_episode = 0, 0
    if args.resume and model_path.exists():
        model = tf.keras.models.load_model(model_path)
        if model.optimizer is None:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                loss=tf.keras.losses.Huber(),
            )
        state = _load_state(state_path) or {}
        start_step = state.get("step", 0)
        start_episode = state.get("episode", 0)
        print(f"Resuming from step {start_step}, episode {start_episode}")
    else:
        model = build_q_network(action_count, input_shape, learning_rate=args.learning_rate)

    target_model = build_q_network(action_count, input_shape, learning_rate=args.learning_rate)
    target_model.set_weights(model.get_weights())
    train_step = _build_train_step(model, target_model, args.gamma, args.clip_norm)

    # Estimate Replay Buffer Memory
    # state = (frames, h, w), next_state same. 2 * capacity * prod(shape) bytes
    # Plus overhead for actions, rewards, etc.
    bytes_per_state = np.prod(input_shape)
    estimated_bytes = args.replay_size * bytes_per_state * 2
    print(f"Initializing Replay Buffer (capacity={args.replay_size}). Estimated state memory: {estimated_bytes / (1024**3):.2f} GB")

    replay = ReplayBuffer(args.replay_size, input_shape, alpha=args.prioritized_alpha)
    replay_lock = threading.Lock()

    # Prefetching setup
    batch_queue = queue.Queue(maxsize=3)
    stop_event = threading.Event()

    def prefetch_worker():
        while not stop_event.is_set():
            with replay_lock:
                size = len(replay)
            
            if size < args.batch_size or size < args.prefetch_steps:
                time.sleep(0.01)
                continue
            
            step = shared_step.value
            beta = _beta_by_step(step, args.beta_start, args.beta_frames)
            
            with replay_lock:
                batch = replay.sample(args.batch_size, beta=beta)
            
            try:
                batch_queue.put(batch, timeout=0.1)
            except queue.Full:
                continue

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    writer = tf.summary.create_file_writer(str(model_dir / "logs"))

    global_step = start_step
    episode = start_episode
    best_reward = -float("inf")
    epsilon = _epsilon_by_step(global_step, args.eps_start, args.eps_final, args.eps_decay)
    start_time = time.time()
    recent_returns = deque(maxlen=ROLLING_WINDOW_SIZE)
    recent_steps = deque(maxlen=ROLLING_WINDOW_SIZE)
    recent_losses = deque(maxlen=ROLLING_WINDOW_SIZE)
    recent_mean_qs = deque(maxlen=ROLLING_WINDOW_SIZE)
    recent_max_qs = deque(maxlen=ROLLING_WINDOW_SIZE)
    recent_grad_norms = deque(maxlen=ROLLING_WINDOW_SIZE)
    last_log_time = time.time()
    last_log_step = global_step
    last_timing_step = global_step
    timing = TimingStats() if args.timing else None

    def train_on_replay(step):
        if step < args.prefetch_steps:
            return
        if step % args.train_freq == 0:
            try:
                t0 = time.perf_counter()
                batch = batch_queue.get(timeout=2.0)
                (
                    batch_states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    indices,
                    weights,
                ) = batch
                
                if timing:
                    timing.add("queue_get", time.perf_counter() - t0)

                t1 = time.perf_counter()
                td_errors, loss, mean_q, max_q, grad_norm = train_step(
                    batch_states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    weights,
                )
                td_errors = td_errors.numpy()
                
                # Update stats
                recent_losses.append(float(loss))
                recent_mean_qs.append(float(mean_q))
                recent_max_qs.append(float(max_q))
                recent_grad_norms.append(float(grad_norm))

                if timing:
                    timing.add("train_step", time.perf_counter() - t1)
                
                with replay_lock:
                    replay.update_priorities(indices, td_errors)
            except queue.Empty:
                pass

    def log_step(step, eps, mean_reward, mean_steps):
        nonlocal last_log_time, last_log_step, last_timing_step
        now = time.time()
        elapsed = now - start_time
        
        avg_loss = np.mean(recent_losses) if recent_losses else 0.0
        avg_mean_q = np.mean(recent_mean_qs) if recent_mean_qs else 0.0
        avg_max_q = np.mean(recent_max_qs) if recent_max_qs else 0.0
        avg_grad_norm = np.mean(recent_grad_norms) if recent_grad_norms else 0.0
        
        # Calculate queue stats
        q_size = transition_queue.qsize()
        with dropped_transitions.get_lock():
            total_drops = dropped_transitions.value
            dropped_transitions.value = 0 # Reset for next interval
            
        time_delta = max(1e-6, now - last_log_time)
        step_delta = max(1, step - last_log_step)
        sps = step_delta / time_delta
        drops_ps = total_drops / time_delta
        
        with writer.as_default():
            tf.summary.scalar("epsilon", eps, step=step)
            tf.summary.scalar("episode_reward_mean", mean_reward, step=step)
            tf.summary.scalar("replay_size", len(replay), step=step)
            tf.summary.scalar("episode_steps_mean", mean_steps, step=step)
            tf.summary.scalar("wall_time_seconds", elapsed, step=step)
            tf.summary.scalar("loss", avg_loss, step=step)
            tf.summary.scalar("q_value_mean", avg_mean_q, step=step)
            tf.summary.scalar("q_value_max", avg_max_q, step=step)
            tf.summary.scalar("grad_norm", avg_grad_norm, step=step)
            tf.summary.scalar("queue_size", q_size, step=step)
            tf.summary.scalar("drops_per_second", drops_ps, step=step)
            
        timing_summary = ""
        if timing and (step - last_timing_step) >= args.timing_interval:
            summary = timing.summary_and_reset()
            if summary:
                timing_summary = f" timing[{summary}]"
            last_timing_step = step
        print(
            f"[step {step}] ep {episode} rw {mean_reward:.2f} len {mean_steps:.1f} "
            f"loss {avg_loss:.4f} q {avg_mean_q:.2f} gn {avg_grad_norm:.2f} "
            f"eps {eps:.3f} sps {sps:.1f} drops {drops_ps:.1f}{timing_summary}",
            flush=True,
        )
        last_log_time = now
        last_log_step = step

    def finalize_episode(ep_return, ep_steps):
        nonlocal best_reward, episode
        if ep_return > best_reward:
            best_reward = ep_return
            if args.save_best:
                best_path = model_dir / "best.keras"
                model.save(best_path, overwrite=True, include_optimizer=False)

        recent_returns.append(ep_return)
        recent_steps.append(ep_steps)
        rolling_mean = float(np.mean(recent_returns)) if recent_returns else ep_return
        elapsed = time.time() - start_time
        with writer.as_default():
            tf.summary.scalar("episode_return", ep_return, step=global_step)
            tf.summary.scalar("episode_return_ma100", rolling_mean, step=global_step)
            tf.summary.scalar("episode_steps", ep_steps, step=global_step)
            tf.summary.scalar("episode_wall_time_seconds", elapsed, step=global_step)

        episode += 1

    # Write initial weights to shared memory and signal actors
    initial_weights = model.get_weights()
    shared_weights.write(initial_weights)
    for conn in conns:
        conn.send(MSG_WEIGHTS_READY)
    shared_step.value = global_step

    def add_transition(trans):
        nonlocal global_step
        state, action, reward, next_state, done = trans
        with replay_lock:
            replay.add(state, action, np.clip(reward, -1.0, 1.0), next_state, done)
        global_step += 1

    while global_step < args.total_steps:
        # Drain all available transitions (non-blocking)
        t_wait = time.perf_counter() if timing else 0
        transitions_added = 0
        while True:
            try:
                add_transition(transition_queue.get_nowait())
                transitions_added += 1
            except queue.Empty:
                break

        # Only block if we need transitions to start training
        if transitions_added == 0 and len(replay) < args.batch_size:
            add_transition(transition_queue.get())
            transitions_added = 1

        if timing and transitions_added > 0:
            timing.add("queue_drain", time.perf_counter() - t_wait)

        shared_step.value = global_step
        epsilon = _epsilon_by_step(global_step, args.eps_start, args.eps_final, args.eps_decay)

        # Sync weights to actors periodically
        if args.actor_sync_interval > 0 and global_step % args.actor_sync_interval == 0:
            t_sync = time.perf_counter() if timing else 0
            shared_weights.write(model.get_weights())
            if timing:
                timing.add("weight_sync", time.perf_counter() - t_sync)

        # Train on replay buffer
        train_on_replay(global_step)
        if global_step % args.target_update == 0 and global_step > 0:
            target_model.set_weights(model.get_weights())

        # Collect episode stats
        while True:
            try:
                ep_return, ep_steps = stats_queue.get_nowait()
            except queue.Empty:
                break
            finalize_episode(ep_return, ep_steps)

        if global_step % args.log_interval == 0:
            mean_reward = float(np.mean(recent_returns)) if recent_returns else 0.0
            mean_steps = float(np.mean(recent_steps)) if recent_steps else 0.0
            log_step(global_step, epsilon, mean_reward, mean_steps)

        if args.save_interval > 0 and global_step % args.save_interval == 0 and global_step > 0:
            model.save(model_path, overwrite=True, include_optimizer=False)
            _save_state(state_path, {
                "step": global_step, "episode": episode,
                "epsilon": float(epsilon), "timestamp": time.time()
            })

    model.save(model_path, overwrite=True, include_optimizer=False)
    _save_state(
        state_path,
        {"step": global_step, "episode": episode, "epsilon": float(epsilon), "timestamp": time.time()},
    )
    # Cleanup actors
    for conn in conns:
        with contextlib.suppress(Exception):
            conn.send(MSG_CLOSE)
        with contextlib.suppress(Exception):
            conn.close()
    for proc in workers:
        proc.join(timeout=PROCESS_JOIN_TIMEOUT)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=PROCESS_JOIN_TIMEOUT)
    with contextlib.suppress(Exception):
        transition_queue.close()
    with contextlib.suppress(Exception):
        stats_queue.close()
    shared_weights.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Double DQN agent for Mario.")

    env = parser.add_argument_group("environment")
    env.add_argument("--env-id", default="SuperMarioBros-v0")
    env.add_argument("--action-set", default="complex", choices=["complex", "simple", "right"])
    env.add_argument("--frame-size", type=int, default=84)
    env.add_argument("--frame-skip", type=int, default=4)
    env.add_argument("--frame-stack", type=int, default=4)
    env.add_argument("--seed", type=int, default=None)

    actors = parser.add_argument_group("parallelism")
    actors.add_argument("--num-actors", type=int, default=8)
    actors.add_argument("--actor-sync-interval", type=int, default=1000)
    actors.add_argument("--actor-queue-size", type=int, default=DEFAULT_QUEUE_SIZE)

    train = parser.add_argument_group("training")
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--clip-norm", type=float, default=1.0)
    train.add_argument("--replay-size", type=int, default=100000)
    train.add_argument("--train-freq", type=int, default=2)
    train.add_argument("--prefetch-steps", type=int, default=0)
    train.add_argument("--target-update", type=int, default=10000)
    train.add_argument("--total-steps", type=int, default=20_000_000)

    explore = parser.add_argument_group("exploration")
    explore.add_argument("--eps-start", type=float, default=1.0)
    explore.add_argument("--eps-final", type=float, default=0.05)
    explore.add_argument("--eps-decay", type=int, default=1_000_000)

    per = parser.add_argument_group("prioritized replay")
    per.add_argument("--prioritized-alpha", type=float, default=0.6)
    per.add_argument("--beta-start", type=float, default=0.4)
    per.add_argument("--beta-frames", type=int, default=200000)

    output = parser.add_argument_group("output")
    output.add_argument("--model-dir", default="models/mario_dqn")
    output.add_argument("--log-interval", type=int, default=25000)
    output.add_argument("--save-interval", type=int, default=100000, help="Save checkpoint every N steps")
    output.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    output.add_argument("--resume", action="store_true")

    perf = parser.add_argument_group("performance")
    perf.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    perf.add_argument("--xla", action=argparse.BooleanOptionalAction, default=True)
    perf.add_argument("--mem-growth", action=argparse.BooleanOptionalAction, default=True)
    perf.add_argument("--timing", action="store_true")
    perf.add_argument("--timing-interval", type=int, default=1000)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_tf(
        use_mixed_precision=args.mixed_precision,
        enable_xla=args.xla,
        enable_mem_growth=args.mem_growth,
    )
    train(args)


if __name__ == "__main__":
    main()
