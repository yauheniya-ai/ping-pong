# train_ppo.py
import numpy as np
import gymnasium as gym
import ale_py
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

plt.style.use("dark_background")


# --------------
# Config
# --------------
CONFIG = {
    "env_id": "ALE/Pong-v5",
    "render_mode": "rgb_array",
    "n_steps": 2048,
    "total_timesteps": 3_000_000,
    "batch_size": 64,
    "update_epochs": 8,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "learning_rate": 2.5e-4,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "save_path": "ppo_pong.keras",
    "log_interval": 1,
    "save_interval": 100_000,   # save plots/checkpoints every 100k steps
    "results_dir": "results",
    "device": "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0",
}


# --------------
# Environment utilities
# --------------
gym.register_envs(ale_py)


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


# --------------
# Model
# --------------
def create_actor_critic(num_actions):
    inp = Input(shape=(80, 80, 4), dtype=tf.float32)
    x = Conv2D(32, 8, strides=4, activation="relu")(inp)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    logits = Dense(num_actions, activation=None, name="logits")(x)
    value = Dense(1, activation=None, name="value")(x)
    return Model(inp, [logits, value])


# --------------
# PPO helpers
# --------------
def compute_gae(rewards, dones, values, last_value, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(T)):
        mask = 0.0 if dones[t] else 1.0
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return returns, advantages


# --------------
# Training loop
# --------------
def train():
    valid_actions = [0, 1, 2, 3, 4, 5]
    num_actions = len(valid_actions)

    env = make_env(render=CONFIG["render_mode"])
    model = create_actor_critic(num_actions)
    optimizer = Adam(learning_rate=CONFIG["learning_rate"], epsilon=1e-5)

    total_timesteps = CONFIG["total_timesteps"]
    n_steps = CONFIG["n_steps"]
    batch_size = CONFIG["batch_size"]
    update_epochs = CONFIG["update_epochs"]

    obs = None
    stacked_frames = None
    ep_returns = []
    ep_lens = []
    ep_reward_acc = 0.0
    ep_len = 0
    total_steps = 0
    start_time = time.time()

    # timestamped results folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG["results_dir"], f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # results storage
    results = []
    last_save = 0  # track last checkpoint step

    # warm reset
    o, info = env.reset()
    cur_frame = preprocess_frame(o)
    state, stacked_frames = stack_frames(None, cur_frame, True)
    obs = state

    while total_steps < total_timesteps:
        mb_states, mb_actions, mb_logprobs, mb_rewards, mb_dones, mb_values = [], [], [], [], [], []

        for step in range(n_steps):
            logits, value = model(obs[None], training=False)
            logits = logits.numpy()[0]
            value = float(value.numpy()[0, 0])
            probs = tf.nn.softmax(logits).numpy()
            action_idx = np.random.choice(num_actions, p=probs)
            action = valid_actions[action_idx]

            o2, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            r_clip = np.clip(r, -1, 1)

            mb_states.append(obs.copy())
            mb_actions.append(action_idx)
            mb_logprobs.append(np.log(probs[action_idx] + 1e-8))
            mb_rewards.append(r_clip)
            mb_dones.append(done)
            mb_values.append(value)

            ep_reward_acc += r
            ep_len += 1
            total_steps += 1

            cur_frame = preprocess_frame(o2)
            obs, stacked_frames = stack_frames(stacked_frames, cur_frame, False)

            if done:
                ep_returns.append(ep_reward_acc)
                ep_lens.append(ep_len)
                o, info = env.reset()
                cur_frame = preprocess_frame(o)
                obs, stacked_frames = stack_frames(None, cur_frame, True)
                ep_reward_acc = 0.0
                ep_len = 0

            if total_steps >= total_timesteps:
                break

        # bootstrap value
        _, last_value = model(obs[None], training=False)
        last_value = float(last_value.numpy()[0, 0])

        mb_states = np.asarray(mb_states, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=np.int32)
        mb_logprobs = np.asarray(mb_logprobs, dtype=np.float32)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool_)
        mb_values = np.asarray(mb_values, dtype=np.float32)

        returns, advantages = compute_gae(mb_rewards, mb_dones, mb_values, last_value, CONFIG["gamma"], CONFIG["gae_lambda"])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_inds = np.arange(len(mb_states))
        for epoch in range(update_epochs):
            np.random.shuffle(batch_inds)
            for start in range(0, len(mb_states), batch_size):
                end = start + batch_size
                mb_idx = batch_inds[start:end]
                states_b = mb_states[mb_idx]
                actions_b = mb_actions[mb_idx]
                old_logprobs_b = mb_logprobs[mb_idx]
                returns_b = returns[mb_idx]
                advs_b = advantages[mb_idx]

                with tf.GradientTape() as tape:
                    logits_b, values_b = model(states_b, training=True)
                    values_b = tf.squeeze(values_b, axis=1)

                    logp_all = tf.nn.log_softmax(logits_b)
                    action_one_hot = tf.one_hot(actions_b, num_actions)
                    new_logprob = tf.reduce_sum(action_one_hot * logp_all, axis=1)

                    ratio = tf.exp(new_logprob - old_logprobs_b)
                    unclipped = ratio * advs_b
                    clipped = tf.clip_by_value(ratio, 1.0 - CONFIG["clip_ratio"], 1.0 + CONFIG["clip_ratio"]) * advs_b
                    policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

                    value_loss = tf.reduce_mean((returns_b - values_b) ** 2) * CONFIG["value_coef"]

                    probs_b = tf.nn.softmax(logits_b)
                    entropy = -tf.reduce_mean(tf.reduce_sum(probs_b * tf.nn.log_softmax(logits_b), axis=1))
                    entropy_loss = -CONFIG["entropy_coef"] * entropy

                    total_loss = policy_loss + value_loss + entropy_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                grads, _ = tf.clip_by_global_norm(grads, CONFIG["max_grad_norm"])
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # --- logging ---
        elapsed = time.time() - start_time
        avg_return = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
        print(
            f"Steps {total_steps}/{total_timesteps} | updates {(total_steps // n_steps)} "
            f"| avg_return(last50) {avg_return:.2f} | elapsed {elapsed/60:.2f}min"
        )
        model.save(CONFIG["save_path"])

        # --- safe checkpointing ---
        if total_steps - last_save >= CONFIG["save_interval"]:
            results.append({"steps": total_steps, "avg_return": avg_return})
            df = pd.DataFrame(results)
            csv_path = os.path.join(run_dir, "training_results.csv")
            df.to_csv(csv_path, index=False)

            plt.figure()
            plt.plot(df["steps"], df["avg_return"], marker="o")
            plt.xlabel("Steps")
            plt.ylabel("Average Return (last 50 eps)")
            plt.title("PPO Pong Learning Curve")
            plt.grid(True)
            plt.savefig(os.path.join(run_dir, f"learning_curve_{total_steps}.png"))
            plt.close()

            model.save(os.path.join(run_dir, f"ppo_pong_{total_steps}.keras"))
            print(f"[Checkpoint] Model and results saved at {total_steps} steps")
            last_save = total_steps

    model.save(CONFIG["save_path"])
    env.close()
    print("Training finished. Model saved to", CONFIG["save_path"])
    print("Results saved to", run_dir)


# --------------
# Evaluation
# --------------
def evaluate(model_path="ppo_pong.keras", episodes=7, render=True):
    env = make_env(render="human" if render else "rgb_array")
    model = tf.keras.models.load_model(model_path, compile=False)

    valid_actions = [0, 1, 2, 3, 4, 5]
    returns = []

    for ep in range(episodes):
        obs, info = env.reset()
        cur_frame = preprocess_frame(obs)
        state, stacked_frames = stack_frames(None, cur_frame, True)
        done = False
        total_reward = 0

        while not done:
            logits, _ = model(state[None], training=False)
            probs = tf.nn.softmax(logits).numpy()[0]
            action_idx = np.argmax(probs)
            action = valid_actions[action_idx]
            obs, r, terminated, truncated, info = env.step(action)
            total_reward += r
            cur_frame = preprocess_frame(obs)
            state, stacked_frames = stack_frames(stacked_frames, cur_frame, False)
            done = terminated or truncated

        returns.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward}")

    env.close()
    print(f"Average reward over {episodes} episodes: {np.mean(returns):.2f}")


if __name__ == "__main__":
    print("Device:", CONFIG["device"])
    train()
    evaluate("ppo_pong.keras", episodes=7, render=True)
