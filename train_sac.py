# train_sac.py
import wandb
import numpy as np
import gymnasium as gym
import ale_py
from PIL import Image
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  

from db import upload_csv

plt.style.use("dark_background")


# --------------
# Config
# --------------
CONFIG = {
    "env_id": "ALE/Pong-v5",
    "render_mode": "rgb_array",
    "model": "SAC",
    "total_timesteps": 3_000_000,
    "buffer_size": 100_000,
    "batch_size": 64,
    "learning_starts": 10_000,
    "gamma": 0.99,
    "tau": 0.005,  # soft update coefficient
    "learning_rate_actor": 3e-4,
    "learning_rate_critic": 3e-4,
    "learning_rate_alpha": 3e-4,
    "target_entropy": None,  # will be set automatically
    "log_interval": 2048,
    "save_interval": 100_000,
    "results_dir": "results",
    "device": "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0",
}

wandb.init(
    project="sac-pong",
    config=CONFIG,
    save_code=False,
    settings=wandb.Settings(disable_git=True)  
)

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
# Models
# --------------
def create_actor(num_actions):
    inp = Input(shape=(80, 80, 4), dtype=tf.float32)
    x = Conv2D(32, 8, strides=4, activation="relu")(inp)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    logits = Dense(num_actions, activation=None)(x)
    return Model(inp, logits)


def create_critic(num_actions):
    inp = Input(shape=(80, 80, 4), dtype=tf.float32)
    x = Conv2D(32, 8, strides=4, activation="relu")(inp)
    x = Conv2D(64, 4, strides=2, activation="relu")(x)
    x = Conv2D(64, 3, strides=1, activation="relu")(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    q_values = Dense(num_actions, activation=None)(x)
    return Model(inp, q_values)


# --------------
# Replay Buffer
# --------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for i in indices:
            s, a, r, ns, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


# --------------
# Training loop
# --------------
def train():
    results_log = []
    results_checkpoint = []
    last_uploaded_idx_log = 0
    last_uploaded_idx_results = 0
    last_save = 0

    valid_actions = [0, 1, 2, 3, 4, 5]
    num_actions = len(valid_actions)

    # Target entropy for automatic temperature tuning
    CONFIG["target_entropy"] = -np.log(1.0 / num_actions) * 0.98

    env = make_env(render=CONFIG["render_mode"])
    
    # Create networks
    actor = create_actor(num_actions)
    critic1 = create_critic(num_actions)
    critic2 = create_critic(num_actions)
    target_critic1 = create_critic(num_actions)
    target_critic2 = create_critic(num_actions)
    
    # Copy weights to target networks
    target_critic1.set_weights(critic1.get_weights())
    target_critic2.set_weights(critic2.get_weights())
    
    # Optimizers
    actor_optimizer = Adam(learning_rate=CONFIG["learning_rate_actor"])
    critic1_optimizer = Adam(learning_rate=CONFIG["learning_rate_critic"])
    critic2_optimizer = Adam(learning_rate=CONFIG["learning_rate_critic"])
    
    # Temperature parameter (alpha) - learnable
    log_alpha = tf.Variable(0.0, dtype=tf.float32)
    alpha_optimizer = Adam(learning_rate=CONFIG["learning_rate_alpha"])
    
    replay_buffer = ReplayBuffer(CONFIG["buffer_size"])
    
    total_timesteps = CONFIG["total_timesteps"]
    batch_size = CONFIG["batch_size"]
    
    stacked_frames = None
    ep_returns = []
    ep_lens = []
    ep_reward_acc = 0.0
    ep_len = 0
    total_steps = 0
    start_time = time.time()
    best_ep_return = -float("inf")

    # timestamped results folder
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG["results_dir"], f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=4)

    # Upload config to NeonDB
    config_kv = pd.DataFrame(list(CONFIG.items()), columns=["key", "value"])
    config_csv_path = os.path.join(run_dir, "config_kv.csv")
    config_kv.to_csv(config_csv_path, index=False)
    upload_csv(run_name=run_id, table_name="config_kv", csv_path=config_csv_path)
    print(f"[INFO] Config uploaded to NeonDB for run {run_id}")

    # warm reset
    o, info = env.reset()
    cur_frame = preprocess_frame(o)
    state, stacked_frames = stack_frames(None, cur_frame, True)

    while total_steps < total_timesteps:
        # Select action
        if total_steps < CONFIG["learning_starts"]:
            # Random exploration
            action_idx = np.random.randint(num_actions)
        else:
            # Sample from policy
            logits = actor(state[None], training=False)
            probs = tf.nn.softmax(logits).numpy()[0]
            action_idx = np.random.choice(num_actions, p=probs)
        
        action = valid_actions[action_idx]
        
        # Step environment
        o2, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        r_clip = np.clip(r, -1, 1)
        
        cur_frame = preprocess_frame(o2)
        next_state, stacked_frames = stack_frames(stacked_frames, cur_frame, False)
        
        # Store transition
        replay_buffer.add(state.copy(), action_idx, r_clip, next_state.copy(), done)
        
        state = next_state
        ep_reward_acc += r
        ep_len += 1
        total_steps += 1
        
        # Train if enough samples
        if total_steps >= CONFIG["learning_starts"] and len(replay_buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            
            alpha = tf.exp(log_alpha)
            
            # Update critics
            with tf.GradientTape(persistent=True) as tape:
                # Current Q-values
                q1_values = critic1(states_b, training=True)
                q2_values = critic2(states_b, training=True)
                
                q1_pred = tf.reduce_sum(q1_values * tf.one_hot(actions_b, num_actions), axis=1)
                q2_pred = tf.reduce_sum(q2_values * tf.one_hot(actions_b, num_actions), axis=1)
                
                # Target Q-values
                next_logits = actor(next_states_b, training=False)
                next_probs = tf.nn.softmax(next_logits)
                next_log_probs = tf.nn.log_softmax(next_logits)
                
                next_q1 = target_critic1(next_states_b, training=False)
                next_q2 = target_critic2(next_states_b, training=False)
                next_q = tf.minimum(next_q1, next_q2)
                
                # Soft value: V = E[Q - α*log(π)]
                next_v = tf.reduce_sum(next_probs * (next_q - alpha * next_log_probs), axis=1)
                
                target_q = rewards_b + CONFIG["gamma"] * (1 - dones_b.astype(np.float32)) * next_v
                
                critic1_loss = tf.reduce_mean((q1_pred - target_q) ** 2)
                critic2_loss = tf.reduce_mean((q2_pred - target_q) ** 2)
            
            grads1 = tape.gradient(critic1_loss, critic1.trainable_variables)
            grads2 = tape.gradient(critic2_loss, critic2.trainable_variables)
            critic1_optimizer.apply_gradients(zip(grads1, critic1.trainable_variables))
            critic2_optimizer.apply_gradients(zip(grads2, critic2.trainable_variables))
            del tape
            
            # Update actor
            with tf.GradientTape() as tape:
                logits = actor(states_b, training=True)
                probs = tf.nn.softmax(logits)
                log_probs = tf.nn.log_softmax(logits)
                
                q1_values = critic1(states_b, training=False)
                q2_values = critic2(states_b, training=False)
                q_values = tf.minimum(q1_values, q2_values)
                
                # Policy loss: maximize E[Q - α*log(π)]
                actor_loss = tf.reduce_mean(tf.reduce_sum(probs * (alpha * log_probs - q_values), axis=1))
            
            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))
            
            # Update temperature
            with tf.GradientTape() as tape:
                logits = actor(states_b, training=False)
                probs = tf.nn.softmax(logits)
                log_probs = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_sum(probs * log_probs, axis=1)
                alpha_loss = -tf.reduce_mean(log_alpha * (entropy - CONFIG["target_entropy"]))
            
            alpha_grads = tape.gradient(alpha_loss, [log_alpha])
            alpha_optimizer.apply_gradients(zip(alpha_grads, [log_alpha]))
            
            # Soft update target networks
            for target, source in [(target_critic1, critic1), (target_critic2, critic2)]:
                for target_var, source_var in zip(target.trainable_variables, source.trainable_variables):
                    target_var.assign(CONFIG["tau"] * source_var + (1 - CONFIG["tau"]) * target_var)
        
        if done:
            ep_returns.append(ep_reward_acc)
            ep_lens.append(ep_len)
            
            # Check best episode
            if ep_reward_acc > best_ep_return:
                best_ep_return = ep_reward_acc
                best_model_path = os.path.join(run_dir, "best.keras")
                actor.save(best_model_path)
                
                best_info = {
                    "episode": len(ep_returns),
                    "steps": total_steps,
                    "reward": ep_reward_acc
                }
                best_csv = os.path.join(run_dir, "best_episode_results.csv")
                pd.DataFrame([best_info]).to_csv(best_csv, index=False)
                upload_csv(run_name=run_id, table_name="best_episode_results", csv_path=best_csv)
                print(f"[Checkpoint] New BEST model (episode={len(ep_returns)}, reward={ep_reward_acc:.2f})")
            
            o, info = env.reset()
            cur_frame = preprocess_frame(o)
            state, stacked_frames = stack_frames(None, cur_frame, True)
            ep_reward_acc = 0.0
            ep_len = 0
        
        # Logging
        if total_steps % CONFIG["log_interval"] == 0 and total_steps >= CONFIG["learning_starts"]:
            elapsed = time.time() - start_time
            avg_return = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            print(f"Steps {total_steps}/{total_timesteps} | avg_return(last50) {avg_return:.2f} | elapsed {elapsed/60:.2f}min")
            
            wandb.log({
                "steps": total_steps,
                "avg_return_last50": avg_return,
                "elapsed_min": elapsed / 60
            })
            
            log_entry = {
                "steps": total_steps,
                "avg_return_last50": round(avg_return, 1),
                "elapsed_min": round(elapsed / 60, 1),
            }
            results_log.append(log_entry)
            log_df = pd.DataFrame(results_log)
            log_path = os.path.join(run_dir, "training_log.csv")
            log_df.to_csv(log_path, index=False)
            
            new_log_rows = log_df.iloc[last_uploaded_idx_log:]
            if not new_log_rows.empty:
                temp_csv = os.path.join(run_dir, "temp_training_log.csv")
                new_log_rows.to_csv(temp_csv, index=False)
                upload_csv(run_name=run_id, table_name="training_log", csv_path=temp_csv)
                last_uploaded_idx_log = len(log_df)
        
        # Checkpointing
        if total_steps - last_save >= CONFIG["save_interval"] and total_steps >= CONFIG["learning_starts"]:
            avg_return = float(np.mean(ep_returns[-50:])) if ep_returns else 0.0
            checkpoint_entry = {"steps": total_steps, "avg_return": avg_return}
            results_checkpoint.append(checkpoint_entry)
            df_results = pd.DataFrame(results_checkpoint)
            csv_path_results = os.path.join(run_dir, "training_results.csv")
            df_results.to_csv(csv_path_results, index=False)
            
            new_results_rows = df_results.iloc[last_uploaded_idx_results:]
            if not new_results_rows.empty:
                temp_csv_results = os.path.join(run_dir, "temp_training_results.csv")
                new_results_rows.to_csv(temp_csv_results, index=False)
                upload_csv(run_name=run_id, table_name="training_results", csv_path=temp_csv_results)
                last_uploaded_idx_results = len(df_results)
            
            plt.figure()
            plt.plot(df_results["steps"], df_results["avg_return"], marker="o")
            plt.xlabel("Steps")
            plt.ylabel("Average Return (last 50 eps)")
            plt.title("SAC Pong Learning Curve")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(run_dir, "learning_curve.png"))
            plt.close()
            
            last_model_path = os.path.join(run_dir, "last.keras")
            actor.save(last_model_path)
            print(f"[Checkpoint] Last model saved at {total_steps} steps")
            last_save = total_steps

    # Final save
    final_path = os.path.join(run_dir, "last.keras")
    actor.save(final_path)
    env.close()
    print(f"Training finished. Last model saved to {final_path}")
    print("Results saved to", run_dir)


# --------------
# Evaluation
# --------------
RESULTS_ROOT = "results"

def get_latest_best_model():
    last_run = sorted([d for d in os.listdir(RESULTS_ROOT) if d.startswith("run_")])[-1]
    return os.path.join(RESULTS_ROOT, last_run, "best.keras")


def evaluate(model_path=None, episodes=7, render=True):
    if model_path is None:
        model_path = get_latest_best_model()
    env = make_env(render="human" if render else "rgb_array")
    actor = tf.keras.models.load_model(model_path, compile=False)
    
    valid_actions = [0, 1, 2, 3, 4, 5]
    returns = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        cur_frame = preprocess_frame(obs)
        state, stacked_frames = stack_frames(None, cur_frame, True)
        done = False
        total_reward = 0
        
        while not done:
            logits = actor(state[None], training=False)
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
    evaluate()