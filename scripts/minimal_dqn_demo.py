# minimal_dqn_demo.py
import gymnasium as gym
import ale_py
import numpy as np
from collections import deque
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
from datetime import datetime

# ---------------------
# Config
# ---------------------
CONFIG = {
    "env_id": "ALE/Pong-v5",
    "render_mode": "rgb_array",
    "total_timesteps": 1_000_000,     # smaller for demo
    "buffer_size": 10_000,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 10_000,
    "train_freq": 4,
    "learning_starts": 1000,
    "device": "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0",
}

gym.register_envs(ale_py)

# ---------------------
# Utils
# ---------------------
def make_env(render="rgb_array"):
    return gym.make(CONFIG["env_id"], render_mode=render)

def preprocess_frame(frame_rgb):
    img = Image.fromarray(frame_rgb)
    img = img.crop((0, 30, 160, 210))
    img = img.convert("L")
    img = img.resize((80, 80), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

def stack_frames(stacked_frames, cur_frame, is_new_episode):
    if is_new_episode or stacked_frames is None:
        stacked_frames = deque([np.zeros((80, 80), dtype=np.float32) for _ in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(cur_frame)
    else:
        stacked_frames.append(cur_frame)
    return np.stack(stacked_frames, axis=-1).astype(np.float32), stacked_frames

# ---------------------
# Q-network
# ---------------------
def create_q_network(num_actions):
    inp = Input(shape=(80, 80, 4), dtype=tf.float32)
    x = Conv2D(32, 8, strides=4, activation="relu")(inp)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    q_values = Dense(num_actions, activation=None)(x)
    return Model(inp, q_values)

# ---------------------
# Replay Buffer
# ---------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, ns, d = zip(*[self.buffer[i] for i in idxs])
        return np.array(s), np.array(a), np.array(r), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)

# ---------------------
# Training Demo
# ---------------------
def train_demo():
    env = make_env()
    valid_actions = [0, 1, 2, 3, 4, 5]
    num_actions = len(valid_actions)

    q_network = create_q_network(num_actions)
    optimizer = Adam(learning_rate=CONFIG["learning_rate"])
    buffer = ReplayBuffer(CONFIG["buffer_size"])

    epsilon_schedule = np.linspace(
        CONFIG["epsilon_start"], CONFIG["epsilon_end"], CONFIG["epsilon_decay_steps"]
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("scripts/demo_results", exist_ok=True)
    data_path = os.path.join("scripts/demo_results", f"transitions_{run_id}.csv")
    data_records = []

    o, info = env.reset()
    cur_frame = preprocess_frame(o)
    state, stacked_frames = stack_frames(None, cur_frame, True)

    total_steps = 0
    episode_reward = 0

    while total_steps < CONFIG["total_timesteps"]:
        # ε-greedy policy
        epsilon = epsilon_schedule[min(total_steps, CONFIG["epsilon_decay_steps"] - 1)]
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            q_values = q_network(state[None], training=False).numpy()[0]
            action_idx = np.argmax(q_values)

        action = valid_actions[action_idx]
        next_obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_frame = preprocess_frame(next_obs)
        next_state, stacked_frames = stack_frames(stacked_frames, next_frame, False)

        # Store transition
        buffer.add(state, action_idx, r, next_state, done)

        # Record a simplified transition sample (for CSV)
        record = {
            "step": total_steps,
            "action": action_idx,
            "reward": r,
            "done": int(done),
            "state_mean": float(np.mean(state)),
            "next_state_mean": float(np.mean(next_state)),
        }
        data_records.append(record)

        # Train if enough data
        if total_steps > CONFIG["learning_starts"] and total_steps % CONFIG["train_freq"] == 0:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(CONFIG["batch_size"])
            next_q = np.max(q_network(ns_b, training=False).numpy(), axis=1)
            target_q = r_b + CONFIG["gamma"] * (1 - d_b) * next_q

            with tf.GradientTape() as tape:
                q_values = q_network(s_b, training=True)
                q_pred = tf.reduce_sum(q_values * tf.one_hot(a_b, num_actions), axis=1)
                loss = tf.reduce_mean(tf.keras.losses.huber(target_q, q_pred))

            grads = tape.gradient(loss, q_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))

        state = next_state
        total_steps += 1
        episode_reward += r

        if done:
            print(f"Episode done at step {total_steps}, reward: {episode_reward:.1f}")
            o, info = env.reset()
            cur_frame = preprocess_frame(o)
            state, stacked_frames = stack_frames(None, cur_frame, True)
            episode_reward = 0

    # Save dataset
    df = pd.DataFrame(data_records)
    df.to_csv(data_path, index=False)
    print(f"\n[INFO] Demo finished. Transitions saved to: {data_path}")
    env.close()

if __name__ == "__main__":
    print("Device:", CONFIG["device"])
    train_demo()
