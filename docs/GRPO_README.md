# GRPO Trainer Explained: Complete Guide

## Table of Contents
1. [What is GRPO?](#what-is-grpo)
2. [Why GRPO Works Better](#why-grpo-works-better)
3. [The Algorithm Step-by-Step](#the-algorithm-step-by-step)
4. [Code Walkthrough](#code-walkthrough)
5. [Interview Q&A](#interview-qa)
6. [Common Pitfalls](#common-pitfalls)

---

## What is GRPO?

**GRPO** (Group Relative Policy Optimization) is a reinforcement learning algorithm for training language models. It's particularly effective when you can generate multiple candidate outputs and compare their quality.

### Core Principle

Instead of asking: **"Is this output good?"**  
GRPO asks: **"Is this output better than the others?"**

This simple shift from absolute to relative comparison dramatically improves training stability and sample efficiency.

### The Key Insight

When training on mathematical proofs:
- **Problem:** Absolute rewards are noisy (a score of 0.7 might be great for hard problems, terrible for easy ones)
- **Solution:** Compare within groups of solutions to the *same* problem
- **Result:** Model learns "what makes one solution better than another" rather than memorizing absolute scores

---

## Why GRPO Works Better

### 1. Reduces Variance

**Traditional RL:**
```
Problem A (easy):   Solutions get rewards [0.9, 0.8, 0.7, 0.6]
Problem B (hard):   Solutions get rewards [0.4, 0.3, 0.2, 0.1]
```
→ Model might learn: "Always avoid hard problems!" ❌

**GRPO:**
```
Problem A: Advantages = [+1.5, +0.5, -0.5, -1.5]  (normalized within group)
Problem B: Advantages = [+1.5, +0.5, -0.5, -1.5]  (same pattern!)
```
→ Model learns: "Generate solutions like the best one in each group" ✓

### 2. Self-Normalizing

GRPO automatically adapts to:
- **Problem difficulty** - Easy and hard problems are treated equally
- **Reward scale** - No manual tuning of reward ranges needed
- **Distribution shift** - As model improves, comparison still works

### 3. Sample Efficient

Each training sample provides K comparisons:
- Traditional: 1 sample = 1 reward signal
- GRPO: K samples = K(K-1)/2 pairwise comparisons
- For K=4: **6x more learning signal** per problem!

---

## The Algorithm Step-by-Step

### High-Level Overview

```
For each training iteration:
  1. Sample N problems
  2. Generate K solutions per problem (N×K total solutions)
  3. Get reward for each solution (via verifier)
  4. Group solutions by problem (N groups of K)
  5. Compute advantages within each group
  6. Update policy to increase probability of high-advantage solutions
```

### Detailed Algorithm

**Step 1: Sample Problems**
```python
problems = sample_n_problems(dataset, n=32)
# ["Prove √2 is irrational", "Find lim x→∞ (1+1/x)^x", ...]
```

**Step 2: Generate Solutions**
```python
for each problem:
    generate K=4 solutions using current policy
    
# Result: 32 problems × 4 solutions = 128 solutions
```

**Step 3: Compute Rewards**
```python
rewards = verifier.score_all_solutions(problems, solutions)
# [0.8, 0.5, 0.5, 0.2,  # Problem 1's 4 solutions
#  1.0, 0.7, 0.3, 0.0,  # Problem 2's 4 solutions
#  ...]
```

**Step 4: Group by Problem**
```python
# Reshape into (N_problems, K_solutions)
rewards_grouped = reshape(rewards, (32, 4))
```

**Step 5: Compute Advantages**
```python
for each group:
    group_mean = mean(group_rewards)
    group_std = std(group_rewards)
    
    for each solution in group:
        advantage = (reward - group_mean) / group_std
```

**Mathematical Formula:**
```
A_i = (R_i - μ_group) / σ_group

where:
  A_i = advantage for solution i
  R_i = reward for solution i
  μ_group = mean reward in the group
  σ_group = standard deviation of rewards in the group
```

**Step 6: Policy Update**
```python
loss = -mean(advantages * log_probabilities)
loss.backward()
optimizer.step()
```

### Concrete Example

**Problem:** "Prove that √2 is irrational"

**Generated 4 solutions:**
1. Perfect proof → Verifier score: **1.0**
2. Minor gap in logic → Verifier score: **0.5**
3. Another minor issue → Verifier score: **0.5**
4. Completely wrong → Verifier score: **0.0**

**Compute group statistics:**
- Mean: (1.0 + 0.5 + 0.5 + 0.0) / 4 = **0.5**
- Std: **0.35**

**Compute advantages:**
- Solution 1: (1.0 - 0.5) / 0.35 = **+1.43** ← Boost this!
- Solution 2: (0.5 - 0.5) / 0.35 = **0.0** ← Neutral
- Solution 3: (0.5 - 0.5) / 0.35 = **0.0** ← Neutral
- Solution 4: (0.0 - 0.5) / 0.35 = **-1.43** ← Suppress this!

**Policy update:**
- Increase probability of generating proofs like Solution 1
- Decrease probability of generating proofs like Solution 4
- Solutions 2 and 3 are average, no strong update

---

## Code Walkthrough

### 1. Initialization

```python
class GRPOTrainer:
    def __init__(
        self,
        model: Any,              # The policy model (LLM)
        reward_fn: Callable,     # Function to compute rewards
        optimizer: Any,          # PyTorch optimizer
        group_size: int = 4,     # K samples per problem
        kl_coef: float = 0.0,    # KL penalty coefficient
        clip_range: float = 0.2  # PPO clipping range
    ):
        self.model = model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.group_size = group_size
        # ...
```

**Key parameters:**
- `group_size`: How many solutions to generate per problem (typically 4-8)
- `kl_coef`: Penalty to prevent model from changing too quickly
- `clip_range`: PPO-style clipping to limit policy updates

### 2. Computing Advantages

```python
def compute_advantages(
    self,
    rewards: List[float],
    group_size: int
) -> Tuple[List[float], Dict[str, float]]:
    """
    Compute advantages using group normalization.
    
    Returns:
        advantages: Normalized rewards within each group
        stats: Mean/std statistics for logging
    """
    # Convert to numpy
    rewards_array = np.array(rewards)
    n_groups = len(rewards) // group_size
    
    # Reshape into groups: (n_groups, group_size)
    rewards_grouped = rewards_array.reshape(n_groups, group_size)
    
    # Compute group statistics
    group_means = rewards_grouped.mean(axis=1, keepdims=True)
    group_stds = rewards_grouped.std(axis=1, keepdims=True)
    
    # Avoid division by zero
    group_stds = np.maximum(group_stds, 1e-8)
    
    # Normalize within groups
    advantages = (rewards_grouped - group_means) / group_stds
    
    return advantages.reshape(-1).tolist()
```

**Why this works:**
- Each group is normalized independently
- Mean advantage per group ≈ 0 (balanced updates)
- Solutions are compared only to peers from same problem

### 3. Policy Loss

```python
def compute_policy_loss(
    self,
    log_probs: torch.Tensor,      # Current policy
    old_log_probs: torch.Tensor,  # Reference policy
    advantages: torch.Tensor       # From compute_advantages()
) -> torch.Tensor:
    """
    Compute policy gradient loss.
    
    Basic formula: loss = -mean(advantages * log_probs)
    
    With PPO clipping: prevents too-large policy updates
    """
    # Importance sampling ratio
    ratio = torch.exp(log_probs - old_log_probs)
    
    # PPO clipping for stability
    ratio_clipped = torch.clamp(
        ratio,
        1.0 - self.clip_range,
        1.0 + self.clip_range
    )
    
    # Take worst case (more conservative)
    policy_loss = -torch.mean(
        torch.min(
            ratio * advantages,
            ratio_clipped * advantages
        )
    )
    
    return policy_loss
```

**PPO Clipping Explained:**
- Prevents catastrophic policy updates
- `ratio = π_new / π_old` (how much policy changed)
- If ratio > 1.2, clip to 1.2 (max 20% increase)
- If ratio < 0.8, clip to 0.8 (max 20% decrease)

### 4. Training Step

```python
def train_step(self, batch: Dict, step: int) -> Dict[str, float]:
    """
    One training step: compute rewards → advantages → loss → update.
    """
    # 1. Compute rewards using verifier
    rewards = self.reward_fn(batch['inputs'], batch['outputs'])
    
    # 2. Compute advantages within groups
    advantages, stats = self.compute_advantages(rewards)
    
    # 3. Compute policy loss
    loss = self.compute_policy_loss(
        batch['log_probs'],
        batch['old_log_probs'],
        torch.tensor(advantages)
    )
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Update weights (if accumulation complete)
    if (step + 1) % self.gradient_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    return stats
```

---

## GENERAL Q&A

### Q1: Why normalize within groups instead of globally?

**A:** Global normalization fails when problems have different difficulties:

```
Global normalization:
  Easy problem solutions: [0.9, 0.8] → advantages: [+1, +0.5]
  Hard problem solutions: [0.3, 0.2] → advantages: [-0.5, -1]
  
  Problem: Model learns to avoid hard problems!
  
Group normalization:
  Easy problem solutions: [0.9, 0.8] → advantages: [+0.7, -0.7]
  Hard problem solutions: [0.3, 0.2] → advantages: [+0.7, -0.7]
  
  Result: Model learns what makes solutions better within each difficulty level.
```

### Q2: What if all solutions in a group have the same reward?

**A:** The standard deviation is 0, so we add epsilon (1e-8) to prevent division by zero. All advantages become 0, meaning no update for this group. This is correct behavior - if all solutions are equally good/bad, there's nothing to learn!

### Q3: How is this different from PPO?

**A:** 
- **PPO**: Uses absolute advantages (compares to value function baseline)
- **GRPO**: Uses *relative* advantages (compares within groups)
- **Both**: Use clipping to prevent large policy updates

GRPO is like "PPO with group-wise normalization" - you can think of it as a specialized variant of PPO.

### Q4: What's the optimal group size?

**A:**
- **Too small (K=2)**: Not enough comparisons, high variance
- **Too large (K=16)**: Expensive, diminishing returns
- **Sweet spot (K=4-8)**: Good balance of comparisons vs. compute

DeepSeekMath-V2 uses K=4 for most training.

### Q5: Can advantages be negative?

**A:** Yes! That's the point. 
- Positive advantage → increase probability
- Negative advantage → decrease probability
- Zero advantage → no change

This is how the model learns to prefer better solutions.

### Q6: How does KL divergence penalty work?

**A:** 
```python
kl_div = E[old_log_probs - new_log_probs]
total_loss = policy_loss + kl_coef * kl_div
```

If `kl_coef > 0`, the model is penalized for changing too much from the reference policy. This prevents:
- Catastrophic forgetting
- Mode collapse
- Extreme policy shifts

### Q7: What's the difference between log_probs and old_log_probs?

**A:**
- `old_log_probs`: From the policy that *generated* the samples (frozen)
- `log_probs`: From the *current* policy being trained (updated)
- `ratio = exp(log_probs - old_log_probs)`: How much the policy changed

This is importance sampling - we generated samples from old policy but are updating new policy.

---

## Common Pitfalls

### 1. Wrong Group Size in Dataset

```python
# ❌ WRONG: Mismatched group size
rewards = [0.8, 0.5, 0.3]  # 3 samples
compute_advantages(rewards, group_size=4)  # Expects 4!

# ✓ CORRECT: Ensure divisibility
assert len(rewards) % group_size == 0
```

### 2. Mixing Solutions from Different Problems

```python
# ❌ WRONG: All solutions in one group
problems = ["P1", "P2", "P3", "P4"]
solutions = ["S1", "S2", "S3", "S4"]
# Each solution is for different problem!

# ✓ CORRECT: Group by problem
problems = ["P1", "P1", "P2", "P2"]  # 2 solutions per problem
solutions = ["S1a", "S1b", "S2a", "S2b"]
```

### 3. Forgetting to Detach old_log_probs

```python
# ❌ WRONG: Gradients flow through old_log_probs
old_log_probs = model.compute_log_probs(tokens)

# ✓ CORRECT: Freeze reference policy
old_log_probs = model.compute_log_probs(tokens).detach()
```

### 4. Not Handling Zero Variance

```python
# ❌ WRONG: Division by zero
group_std = rewards.std()
advantages = (rewards - rewards.mean()) / group_std  # Crash if std=0!

# ✓ CORRECT: Add epsilon
group_std = np.maximum(rewards.std(), 1e-8)
```

### 5. Ignoring Gradient Accumulation

```python
# ❌ WRONG: Update every step with small batches
for batch in dataloader:
    loss = compute_loss(batch)
    loss.backward()
    optimizer.step()  # Too many updates!

# ✓ CORRECT: Accumulate gradients
for step, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## Key Takeaways

1. **GRPO = Relative comparison within groups** - More stable than absolute rewards
2. **Group size (K) is crucial** - Typically 4-8 for good balance
3. **Advantages are normalized** - Mean ≈ 0 within each group
4. **Works with PPO clipping** - Prevents catastrophic policy updates
5. **Sample efficient** - K samples give O(K²) comparisons
6. **Self-normalizing** - Adapts to problem difficulty automatically