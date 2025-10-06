# Ping Pong Training

This projectd tests and compares a wide range of Reinforcement Learning (RL) models to train agents that play Ping Pong. The Arcade Learning Environment (ALE) is a widely recognized framework that provides a challenging and diverse set of Atari 2600 games, enabling the development and evaluation of AI agents through interaction with raw pixel inputs, discrete actions, and game scores in a controlled emulated environment (Fig. 1). In the visualization, the orange paddle represents the built-in opponent, while the green paddle represents the trained agent.

<p align="center">
  <img src="./im/best.gif" alt="Pong Training" width="350" />
  <br><em>Fig. 1: Deep reinforcement learning agent<br> training on Atari Pong using PPO.</em>
</p>


## Project Overview

This project implements and compares various reinforcement learning agents that learn to play Atari Pong from raw pixel inputs. Key features include frame preprocessing, frame stacking for motion capture, convolutional neural networks for visual processing, and reward shaping to encourage effective behaviors.

## Models Overview

| Abbreviation | Name                                          | Key Features                                                                                          |
|--------------|-----------------------------------------------|-----------------------------------------------------------------------------------------------------|
| DQN          | Deep Q-Network                                | Value-based, uses CNN with Q-learning, experience replay, target networks, good for discrete action spaces |
| PPO          | Proximal Policy Optimization                  | On-policy, robust and stable policy gradients, uses clipping to constrain policy updates, widely used baseline |
| SAC          | Soft Actor-Critic                             | Off-policy, actor-critic method, encourages exploration via entropy maximization, works well in continuous spaces |
| A3C          | Asynchronous Advantage Actor-Critic           | Multiple agents in parallel, combines policy and value learning, improves sample efficiency and exploration |
| TD3          | Twin Delayed Deep Deterministic Policy Gradients | Off-policy, addresses overestimation in DDPG, uses two critics, target smoothing, suited for continuous actions |
| DDPG         | Deep Deterministic Policy Gradient            | Off-policy, suitable for continuous control, deterministic actor-critic, experience replay, requires careful tuning |
| HER          | Hindsight Experience Replay                   | Re-labels failed experiences as successes for sparse rewards, often combined with DDPG/TD3           |
| HRL          | Hierarchical Reinforcement Learning           | Incorporates hierarchies of policies or skills, enables learning complex behaviors like strategy and tactics |
| IRL          | Inverse Reinforcement Learning                 | Learns reward functions from expert demonstrations, useful for mimicking real players or strategies  |
| APRG         | Actor-Parametrized Reward Gradient             | Deterministic policy gradient, sample efficient, robust for noisy real-world table tennis settings    |



## Requirements

```bash
pip install -r requirements.txt
```

## Environment Setup

Rename `.env.example` to `.env` and update the environment variables before running the training scripts.

## Usage

Currently implemented algorithms include Deep Q-Network (DQN), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC).

```bash
python train_dqn.py
```

```bash
python train_ppo.py
```

```bash
python train_sac.py
```

## File Structure

```
.
├── train_dqn.py            # Training script using DQN
├── train_ppo.py            # Training script using PPO
├── train_sac.py            # Training script using SAC
├── db.py                   # NeonDB upload setup
├── requirements.txt        # Dependenices
├── .env.example            # Env variables for remote monitoring
├── results/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── best.keras                  # Model checkpoint at highest return
│       ├── last.keras                  # Model checkpoint at last save interval
│       ├── best_episode_results.csv    # Highest reward so far in an episode
│       ├── training_results.csv        # Log ∅ return every 100 000 steps 
│       ├── training_log.csv            # Log ∅ return, elapsed time every 2048 steps 
│       ├── config_kv.csv               # configuration key-value pair
│       └── learning_curve.png          # Plot ∅ return vs. training steps
├── studies/                # References for this project
├── models/                 # Notes for each model (DQN, PPO, SAC, etc.)
├── im/                     # Images for this project
└── README.md               # This document
```

## Neural Network
```
Input (80x80x4) 
  → Conv2D(32, 8x8, stride=4) + ReLU
  → Conv2D(64, 4x4, stride=2) + ReLU  
  → Conv2D(64, 3x3, stride=1) + ReLU
  → Flatten
  → Dense(512) + ReLU
  → Policy Head: Dense(6) [action logits]
  → Value Head: Dense(1) [state value]
```

## Action Space 

Each action corresponds to a paddle move in the *ALE/Pong-v5* environment, e.g. `valid_actions = [0, 1, 2, 3, 4, 5]` represent a discrete(6) set of actions → {NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE}

All Atari joystick + fire button combinations: Discrete(18)  → {NOOP, FIRE, UP, RIGHT, LEFT, DOWN, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT, UPFIRE, RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE}

*ALE/Pong-v5* does not support continuous actions, since it mimics the Atari 2600’s digital controller.


## Results

Results are saved to `results/run_<timestamp>/` including:

- `config.json` specifies the configuration.
- `training_results.csv` saves the logged training progress as a table with the total steps and corresponding average return values.
- `learning_curve.png` visualizes this progress, showing how the agent’s average return evolves over training steps (Fig. 2, Fig. 3).
- `last.keras` and `best.keras` model checkpoints

<div align="center">
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <td style="border: none; text-align: center; vertical-align: top; padding: 0 8px;">
        <img src="./im/learning_curve_ppo.png" alt="Pong Training" width="350" /><br>
        <em>Fig. 2: PPO Training Progress.</em>
      </td>
      <td style="border: none; text-align: center; vertical-align: top; padding: 0 8px;">
        <img src="./im/learning_curve_dqn.png" alt="Other Training" width="350" /><br>
        <em>Fig. 3: DQN Training Progress.</em>
      </td>
    </tr>
  </table>
</div>



## Evaluation

To test the trained PPO agent, navigate to the results of the latest run and execute the evaluate function, for example:

```bash
python -c "import train_ppo; train_ppo.evaluate('results/run_20251004_002148/best.keras')"
```

## Monitoring

Training RL models for over 3 million steps takes several hours to days, making remote monitoring of progress essential. For this project, I have set up [Weights & Biases (wandb)](https://wandb.ai) monitoring initially, and later implemented [custom monitoring](https://rl-dashboard-frontend.onrender.com) using NeonDB and Render (Fig. 4).

<p align="center">
  <img src="./im/NeonDB_Render.png" alt="Remote Monitoring" width="350" />
  <br><em>Fig. 4: Remote Monitoring using NeonDB and Render.</em>
</p>

## Troubleshooting

**Training is slow**: 
- Ensure TensorFlow is using GPU: `tf.config.list_physical_devices('GPU')`
- Reduce `total_timesteps` or `n_steps`

**Agent not learning**:
- Check that rewards are being received (print episode returns)
- Verify frame preprocessing is working correctly
- Try adjusting learning rate or entropy coefficient

**Memory issues**:
- Reduce `n_steps` or `batch_size`
- Use mixed precision training

## Acknowledgments

- OpenAI Gym/Gymnasium for the Atari environment

## License

MIT