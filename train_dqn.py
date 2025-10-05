# train_dqn.py
import wandb
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
    "model": "DQN",
    "total_timesteps": 3_000_000,
    "buffer_size": 100_000,
    "batch_size": 32,
    "learning_starts": 50_000,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 1_000_000,
    "target_update_freq": 10_000,
    "train_freq": 4,
    "log_interval": 2048,
    "save_interval": 100_000,
    "results_dir": "results",
    "device": "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0",
}

wandb.init(
    project="dqn-pong",
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
# DQN Model
# --------------
def create_q_network(num_actions):
    """Create Q-network (Nature DQN architecture)"""
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

    env = make_env(render=CONFIG["render_mode"])
    
    # Create Q-network and target network
    q_network = create_q_network(num_actions)
    target_network = create_q_network(num_actions)
    target_network.set_weights(q_network.get_weights())
    
    optimizer = Adam(learning_rate=CONFIG["learning_rate"])
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
    
    # Epsilon decay schedule
    epsilon_schedule = np.linspace(
        CONFIG["epsilon_start"],
        CONFIG["epsilon_end"],
        CONFIG["epsilon_decay_steps"]
    )

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
        # Epsilon-greedy action selection
        if total_steps < CONFIG["learning_starts"]:
            # Random exploration
            action_idx = np.random.randint(num_actions)
        else:
            epsilon = epsilon_schedule[min(total_steps, CONFIG["epsilon_decay_steps"] - 1)]
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(num_actions)
            else:
                q_values = q_network(state[None], training=False).numpy()[0]
                action_idx = np.argmax(q_values)
        
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
        
        # Train every train_freq steps
        if (total_steps >= CONFIG["learning_starts"] and 
            len(replay_buffer) >= batch_size and 
            total_steps % CONFIG["train_freq"] == 0):
            
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)
            
            # Compute target Q-values
            next_q_values = target_network(next_states_b, training=False).numpy()
            max_next_q = np.max(next_q_values, axis=1)
            target_q = rewards_b + CONFIG["gamma"] * (1 - dones_b.astype(np.float32)) * max_next_q
            
            # Update Q-network
            with tf.GradientTape() as tape:
                q_values = q_network(states_b, training=True)
                action_masks = tf.one_hot(actions_b, num_actions)
                q_pred = tf.reduce_sum(q_values * action_masks, axis=1)
                
                # Huber loss (less sensitive to outliers than MSE)
                loss = tf.reduce_mean(tf.keras.losses.huber(target_q, q_pred))
            
            grads = tape.gradient(loss, q_network.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 10.0)
            optimizer.apply_gradients(zip(grads, q_network.trainable_variables))
        
        # Update target network
        if total_steps % CONFIG["target_update_freq"] == 0:
            target_network.set_weights(q_network.get_weights())
            print(f"[INFO] Target network updated at step {total_steps}")
        
        if done:
            ep_returns.append(ep_reward_acc)
            ep_lens.append(ep_len)
            
            # Check best episode
            if ep_reward_acc > best_ep_return:
                best_ep_return = ep_reward_acc
                best_model_path = os.path.join(run_dir, "best.keras")
                q_network.save(best_model_path)
                
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
            epsilon = epsilon_schedule[min(total_steps, CONFIG["epsilon_decay_steps"] - 1)]
            
            print(f"Steps {total_steps}/{total_timesteps} | avg_return(last50) {avg_return:.2f} | "
                  f"epsilon {epsilon:.3f} | elapsed {elapsed/60:.2f}min")
            
            wandb.log({
                "steps": total_steps,
                "avg_return_last50": avg_return,
                "epsilon": epsilon,
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
            plt.title("DQN Pong Learning Curve")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(run_dir, "learning_curve.png"))
            plt.close()
            
            last_model_path = os.path.join(run_dir, "last.keras")
            q_network.save(last_model_path)
            print(f"[Checkpoint] Last model saved at {total_steps} steps")
            last_save = total_steps

    # Final save
    final_path = os.path.join(run_dir, "last.keras")
    q_network.save(final_path)
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
    q_network = tf.keras.models.load_model(model_path, compile=False)
    
    valid_actions = [0, 1, 2, 3, 4, 5]
    returns = []
    
    for ep in range(episodes):
        obs, info = env.reset()
        cur_frame = preprocess_frame(obs)
        state, stacked_frames = stack_frames(None, cur_frame, True)
        done = False
        total_reward = 0
        
        while not done:
            q_values = q_network(state[None], training=False).numpy()[0]
            action_idx = np.argmax(q_values)
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