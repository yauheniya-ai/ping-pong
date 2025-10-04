# Soft Actor-Critic (SAC): A Deep Reinforcement Learning Algorithm for Continuous Control

Mnih et al. (2013), in their groundbreaking paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.560), laid the foundation for value-based deep RL by combining deep neural networks and Q-learning to learn control policies directly from raw pixels, marking a milestone in applying deep learning to reinforcement learning.

Haarnoja et al. (2018) in the paper ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290) introduced Soft Actor-Critic (SAC), an off-policy actor-critic algorithm based on the maximum entropy reinforcement learning framework. SAC achieves state-of-the-art performance on continuous control benchmarks while demonstrating improved sample efficiency and stability compared to on-policy methods like PPO.

The algorithm was further refined in Haarnoja et al. (2018) ["Soft Actor-Critic Algorithms and Applications"](https://arxiv.org/abs/1812.05905), which introduced automatic temperature tuning and demonstrated SAC's effectiveness on robotic manipulation tasks.

Fujimoto et al. (2018) addressed critical function approximation errors in actor-critic methods in ["Addressing Function Approximation Error in Actor-Critic Methods"](https://arxiv.org/abs/1802.09477), proposing the TD3 algorithm that uses twin critics and delayed policy updates to reduce overestimation bias and improve learning stability in continuous control environments.

## Core Concept

SAC combines three key ideas:
1. **Actor-Critic Architecture**: Separate networks for policy (actor) and value estimation (critics)
2. **Off-Policy Learning**: Learn from past experiences stored in a replay buffer
3. **Maximum Entropy**: Encourage exploration by maximizing both reward and policy entropy

The maximum entropy objective balances exploitation (getting high rewards) with exploration (maintaining randomness in actions), leading to more robust policies.

## Key Components

### 1. The Actor (Policy π)

The actor is a stochastic policy that outputs a probability distribution over actions:

```
π(a|s) = probability of taking action a in state s
```

For discrete actions (like Pong), this is implemented as a softmax over action logits.

### 2. Twin Critics (Q-functions)

SAC uses two Q-networks (Q₁ and Q₂) to estimate action values. This addresses the overestimation bias present in single-critic methods:

```
Q(s,a) = expected discounted return from taking action a in state s
```

The twin critics are updated independently, and the minimum of the two is used for policy updates.

### 3. Target Networks

To stabilize training, SAC maintains target versions of both critics (Q̄₁ and Q̄₂) that are slowly updated using exponential moving averages.

### 4. Temperature Parameter (α)

The temperature parameter controls the exploration-exploitation tradeoff. SAC learns this parameter automatically by optimizing:

```
α* = arg min E[α(-log π(a|s) - H̄)]
```

Where H̄ is the target entropy.

## The SAC Algorithm

### Step 1: Collect Experience

Unlike on-policy methods, SAC stores all experiences in a replay buffer:
- States (game screens)
- Actions taken
- Rewards received
- Next states
- Done flags

The agent can sample random minibatches from this buffer for training, enabling off-policy learning.

### Step 2: Update Critics (Q-Networks)

The critics are trained to minimize the soft Bellman residual. For each critic Qᵢ (i = 1, 2):

```
L^Q = E[(Qᵢ(s,a) - (r + γ(1-d)·V̄(s')))²]
```

Where the soft value function V̄ is computed using the target critics:

```
V̄(s') = E[min(Q̄₁(s',a'), Q̄₂(s',a')) - α·log π(a'|s')]
```

The minimum operation between the twin critics reduces overestimation. The entropy term (α·log π) encourages maintaining action diversity.

### Step 3: Update Actor (Policy)

The actor is trained to maximize the expected soft Q-value:

```
L^π = E[α·log π(a|s) - min(Q₁(s,a), Q₂(s,a))]
```

This objective encourages the policy to:
1. Choose actions with high Q-values (exploitation)
2. Maintain high entropy (exploration via the α·log π term)

### Step 4: Update Temperature (α)

The temperature parameter is learned automatically by minimizing:

```
L^α = E[-α(log π(a|s) + H̄)]
```

Where H̄ is the target entropy (typically set to -log(1/|A|) for discrete actions, where |A| is the number of actions).

### Step 5: Soft Update Target Networks

Target networks are updated using exponential moving averages with coefficient τ (typically 0.005):

```
Q̄ᵢ ← τ·Qᵢ + (1-τ)·Q̄ᵢ
```

This slow update provides stable learning targets.

## Complete SAC Objective

The total loss combines all three components:

```
L^TOTAL = L^Q₁ + L^Q₂ + L^π + L^α
```

Each component is optimized with separate Adam optimizers, typically with learning rate 3e-4.

## Why SAC Works

### 1. Sample Efficiency
Off-policy learning with replay buffers allows each experience to be reused multiple times, dramatically improving sample efficiency compared to on-policy methods.

### 2. Stability
- **Twin critics** reduce overestimation bias
- **Target networks** provide stable learning targets
- **Soft updates** prevent catastrophic forgetting

### 3. Exploration
The maximum entropy framework naturally encourages exploration without requiring separate exploration strategies (like ε-greedy). The learned temperature parameter automatically adjusts exploration over training.

### 4. Robustness
SAC's stochastic policies are more robust to perturbations and generalize better to unseen situations compared to deterministic policies.

## Hyperparameters for Pong

```python
CONFIG = {
    "buffer_size": 100_000,           # Replay buffer capacity
    "batch_size": 64,                 # Minibatch size
    "learning_starts": 10_000,        # Random exploration before training
    "gamma": 0.99,                    # Discount factor
    "tau": 0.005,                     # Target network update coefficient
    "learning_rate_actor": 3e-4,      # Actor learning rate
    "learning_rate_critic": 3e-4,     # Critic learning rate
    "learning_rate_alpha": 3e-4,      # Temperature learning rate
    "target_entropy": -log(1/6)*0.98  # Target entropy (98% of max)
}
```

## Key Differences from PPO

| Aspect | PPO | SAC |
|--------|-----|-----|
| **Learning** | On-policy (uses fresh data) | Off-policy (reuses old data) |
| **Sample Efficiency** | Lower | Higher |
| **Exploration** | Entropy bonus | Maximum entropy objective |
| **Stability** | Clipped updates | Twin critics + target networks |
| **Value Function** | Single V(s) | Twin Q(s,a) functions |
| **Best For** | Stable, simple tasks | Sample-limited, complex tasks |

## References

Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1802.09477

Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. *International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/1801.01290

Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., Kumar, V., Zhu, H., Gupta, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic Algorithms and Applications. *arXiv preprint*. https://arxiv.org/abs/1812.05905

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing Atari with Deep Reinforcement Learning. *arXiv preprint*. https://arxiv.org/abs/1312.5602
