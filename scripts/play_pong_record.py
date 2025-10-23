# play_pong_and_record.py
import gymnasium as gym
import ale_py
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import os

# ---------------------
# Config
# ---------------------
MODEL_PATH = "results/run_20251005_103336/last.keras"
OUTPUT_VIDEO = "scripts/play_pong_dqn.mp4"

# ---------------------
# Environment setup
# ---------------------
gym.register_envs(ale_py)

def make_env():
    return gym.make("ALE/Pong-v5", render_mode="rgb_array")

def preprocess_frame(frame_rgb):
    """Convert raw Atari frame to 80x80 grayscale normalized float."""
    img = Image.fromarray(frame_rgb)
    img = img.crop((0, 30, 160, 210))
    img = img.convert("L")
    img = img.resize((80, 80), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0

# ---------------------
# Frame stacking
# ---------------------
from collections import deque
def stack_frames(stacked_frames, cur_frame, is_new_episode):
    if is_new_episode or stacked_frames is None:
        stacked_frames = deque([np.zeros((80, 80), dtype=np.float32) for _ in range(4)], maxlen=4)
        for _ in range(4):
            stacked_frames.append(cur_frame)
    else:
        stacked_frames.append(cur_frame)
    return np.stack(stacked_frames, axis=-1).astype(np.float32), stacked_frames

# ---------------------
# Play one episode and save to MP4
# ---------------------
def play_and_record():
    env = make_env()
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    valid_actions = [0, 1, 2, 3, 4, 5]

    obs, info = env.reset()
    cur_frame = preprocess_frame(obs)
    state, stacked_frames = stack_frames(None, cur_frame, True)
    done = False

    frames = []
    total_reward = 0

    print(f"[INFO] Loaded model from: {MODEL_PATH}")
    print("[INFO] Playing one episode...")

    while not done:
        # Choose greedy action
        q_values = model(state[None], training=False).numpy()[0]
        action_idx = np.argmax(q_values)
        action = valid_actions[action_idx]

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        # Save rendered frame for video
        frame_rgb = env.render()
        frames.append(frame_rgb)

        # Update state
        next_frame = preprocess_frame(obs)
        state, stacked_frames = stack_frames(stacked_frames, next_frame, False)

    env.close()
    print(f"[INFO] Episode finished. Total reward: {total_reward:.1f}")
    print(f"[INFO] Saving {len(frames)} frames to {OUTPUT_VIDEO} ...")

    # ---------------------
    # Save frames to MP4
    # ---------------------
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,  # FPS
        (width, height)
    )
    for frame in frames:
        # Convert RGB -> BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    print(f"[INFO] Video saved as: {os.path.abspath(OUTPUT_VIDEO)}")

if __name__ == "__main__":
    play_and_record()
