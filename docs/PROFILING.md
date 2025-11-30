# Profiling & Metrics - Explained

## What We Built (75 words)

**Profiling & Metrics layer** = Monitoring infrastructure for training

**profiling.py:**
- GPU memory tracking (detect leaks, peak usage)
- Timing (forward/backward/optimizer)
- Throughput (samples/sec, tokens/sec)

**metrics.py:**
- Training metrics (loss, reward, scores)
- Statistics (mean, std, percentiles)
- JSON export for analysis

**40_profile_gpu.py:**
- Standalone GPU benchmarking
- Memory capacity testing
- Model compatibility check

Think: Instrumentation panel for your training car.

---

## Why We Need This (Before Training!)

**Problem:** Training can fail silently or inefficiently:
- ❌ OOM (Out Of Memory) crashes after 2 hours
- ❌ Gradual memory leaks eating VRAM
- ❌ Slow throughput wasting GPU time
- ❌ Can't tell if training is progressing

**Solution:** Monitor EVERYTHING:
- ✅ Memory usage → Detect leaks early
- ✅ Timing → Find bottlenecks
- ✅ Throughput → Optimize batch size
- ✅ Metrics → Track convergence

**Analogy:** Like a car dashboard - you need gauges BEFORE driving, not after the engine explodes!

---

## profiling.py - Deep Dive

### **1. GPUMemoryProfiler**

**What it tracks:**
```python
profiler = GPUMemoryProfiler(device=0)

# Take measurement
stats = profiler.measure("after_forward")
# {
#   'allocated_gb': 12.5,  # Actually used
#   'reserved_gb': 14.0,   # Reserved by PyTorch
#   'max_allocated_gb': 15.2,  # Peak usage
#   'max_reserved_gb': 16.0
# }
```

**Key methods:**

**`set_baseline()`** - Mark current memory as baseline
```python
profiler.set_baseline()  # After model load
# ... training ...
leak = profiler.detect_leak(threshold_gb=0.5)
if leak:
    print("⚠ Memory leak detected!")
```

**Why useful?** Catch memory leaks DURING training, not after crash!

---

### **2. TrainingTimer**

**What it tracks:**
```python
timer = TrainingTimer(device=0)

timer.start("forward")
output = model(input)
timer.stop("forward")  # Returns: 0.125 sec

timer.start("backward")
loss.backward()
timer.stop("backward")  # Returns: 0.250 sec
```

**Summary:**
```python
summary = timer.get_summary()
# {
#   'forward': {'mean_sec': 0.125, 'count': 100, ...},
#   'backward': {'mean_sec': 0.250, 'count': 100, ...},
#   'optimizer': {'mean_sec': 0.050, 'count': 100, ...}
# }
```

**Why useful?** Find slowest operation → optimize it first!

---

### **3. ThroughputTracker**

**What it tracks:**
```python
tracker = ThroughputTracker(window_size=100)

for batch in dataloader:
    # Process batch...
    tracker.update(
        num_samples=batch_size,
        num_tokens=total_tokens
    )

throughput = tracker.get_current_throughput()
# {
#   'samples_per_sec': 8.5,
#   'tokens_per_sec': 10920
# }
```

**Why useful?** Measure training efficiency - faster = less $$$ on GPU!

---

### **4. ComprehensiveProfiler**

**All-in-one profiler:**
```python
profiler = ComprehensiveProfiler(device=0)

for epoch in range(num_epochs):
    for batch in dataloader:
        profiler.start_iteration()
        
        # Training step...
        
        profiler.end_iteration(
            num_samples=batch_size,
            num_tokens=total_tokens
        )

# Export everything
profiler.export_to_json("profiling_results.json")
```

**Exports:**
```json
{
  "memory": {
    "peak_allocated_gb": 15.2,
    "mean_allocated_gb": 12.8
  },
  "timing": {
    "iteration": {"mean_sec": 0.425}
  },
  "throughput": {
    "samples_per_sec": 9.4
  }
}
```

---

## metrics.py - Deep Dive

### **1. MetricTracker**

**Basic usage:**
```python
tracker = MetricTracker()

for step in range(1000):
    loss = train_step()
    tracker.update("loss", loss, step=step)

# Get statistics
print(tracker.get_mean("loss"))  # 0.342
print(tracker.get_std("loss"))   # 0.089

summary = tracker.get_summary("loss")
# {
#   'count': 1000,
#   'mean': 0.342,
#   'std': 0.089,
#   'min': 0.123,
#   'max': 0.789,
#   'p95': 0.512
# }
```

**Batch updates:**
```python
tracker.update_batch({
    'loss': 0.5,
    'reward': 0.8,
    'kl_div': 0.01
}, step=100)
```

---

### **2. GRPOMetricsTracker**

**Specialized for GRPO training:**
```python
grpo_tracker = GRPOMetricsTracker()

grpo_tracker.update_grpo_step(
    step=100,
    policy_loss=0.45,
    mean_reward=0.72,
    reward_components={
        'format': 1.0,
        'score': 0.68,
        'meta': 0.85
    },
    kl_divergence=0.012
)

# Store group rewards for analysis
grpo_tracker.update_group_rewards([0.9, 0.7, 0.5, 0.3])

# Analyze distribution
dist = grpo_tracker.get_reward_distribution()
# {
#   'reward_mean': 0.72,
#   'reward_std': 0.18,
#   'rank_distribution': {...}
# }
```

**Why specialized?** GRPO needs group-wise analysis, not just batch means!

---

### **3. ScoreDistributionTracker**

**Track verification scores:**
```python
score_tracker = ScoreDistributionTracker()

# Batch of verification scores
scores = [1.0, 0.5, None, 1.0, 0.0]  # None = invalid
score_tracker.update(scores)

dist = score_tracker.get_distribution()
# {
#   'total': 5,
#   'valid_count': 4,
#   'invalid_count': 1,
#   'invalid_rate': 0.2,
#   'score_counts': {'0.0': 1, '0.5': 1, '1.0': 2},
#   'mean_score': 0.625
# }
```

**Why track scores separately?** Monitor verifier quality over time!

---

### **4. TrainingLogger**

**Complete logging solution:**
```python
logger = TrainingLogger(
    log_dir="experiments/logs",
    experiment_name="verifier_iter1"
)

for step in range(1000):
    # Training...
    
    logger.log_step(
        step=step,
        metrics={'loss': loss, 'reward': reward},
        scores=[1.0, 0.5, 1.0, 0.0]
    )
    
    # Save at checkpoints
    if step % 100 == 0:
        logger.save_checkpoint_metrics(f"step_{step}")

# Final export
logger.export_all()
```

**Exports:**
```json
{
  "experiment_name": "verifier_iter1",
  "metric_summaries": {
    "loss": {...},
    "reward": {...}
  },
  "score_distribution": {
    "mean_score": 0.78,
    "score_counts": {...}
  },
  "raw_metrics": {
    "loss": [0.5, 0.48, 0.45, ...]
  }
}
```

---

## 40_profile_gpu.py - Benchmarking Script

### **What it does:**

1. **GPU Info** - Lists available GPUs
2. **Memory Profiling** - Current usage
3. **Model Requirements** - Estimates for different models
4. **Throughput Benchmark** - Mock performance test
5. **Compatibility Check** - Which models fit?

### **Usage:**

```bash
# Run profiling
python scripts/40_profile_gpu.py

# Specify device and output
python scripts/40_profile_gpu.py --device 0 --output results.json

# Skip throughput (faster)
python scripts/40_profile_gpu.py --skip-throughput
```

### **Example Output:**

```
============================================================
Model Memory Requirements
============================================================

Model                     Inference (fp16)     Training (fp16+Adam)     
----------------------------------------------------------------------
DeepSeek-Math-7B            13.0 GB              78.2 GB
DeepSeek-Math-70B          130.4 GB             782.3 GB
DeepSeek-V3 (active)        68.9 GB             413.5 GB

============================================================
Model Compatibility Check
============================================================

Available GPU Memory: 16.00 GB

Model                             Required        Status         
-----------------------------------------------------------------
DeepSeek-Math-7B (inference)       14.0 GB       ✓ Fits
DeepSeek-Math-7B (training)        60.0 GB       ✗ Too large
DeepSeek-V3 active (training)     296.0 GB       ✗ Too large
```

**Verdict for T4 (16GB):**
- ✅ 7B inference: Fits!
- ❌ 7B training: Need LoRA/QLoRA
- ❌ V3 anything: Impossible

---

## JSON Export Format

### **Profiling JSON:**
```json
{
  "timestamp": "2025-11-30T12:00:00",
  "device": 0,
  "memory": {
    "peak_allocated_gb": 15.2,
    "mean_allocated_gb": 12.8,
    "baseline_gb": 10.5
  },
  "timing": {
    "iteration": {
      "mean_sec": 0.425,
      "count": 1000
    }
  },
  "throughput": {
    "samples_per_sec": 9.4,
    "tokens_per_sec": 12032
  }
}
```

### **Metrics JSON:**
```json
{
  "experiment_name": "verifier_training",
  "metric_summaries": {
    "loss": {
      "mean": 0.342,
      "std": 0.089,
      "min": 0.123,
      "max": 0.789
    }
  },
  "score_distribution": {
    "mean_score": 0.78,
    "score_counts": {"0.0": 45, "0.5": 123, "1.0": 832}
  }
}
```

**Why JSON?**
- ✅ Language-agnostic (analyze in Python/R/JS)
- ✅ Easy to parse and visualize
- ✅ Human-readable
- ✅ Version control friendly

---

## Integration Example

**Using profiling + metrics together:**

```python
from src.utils.profiling import ComprehensiveProfiler
from src.utils.metrics import TrainingLogger

# Setup
profiler = ComprehensiveProfiler(device=0)
logger = TrainingLogger("experiments/logs", "verifier_iter1")

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        # Start profiling
        profiler.start_iteration()
        
        # Training step
        loss, reward, scores = training_step(batch)
        
        # End profiling
        profiler.end_iteration(
            num_samples=len(batch),
            num_tokens=batch.total_tokens
        )
        
        # Log metrics
        logger.log_step(
            step=step,
            metrics={'loss': loss, 'reward': reward},
            scores=scores
        )
    
    # Export at end of epoch
    profiler.export_to_json(f"profiling_epoch{epoch}.json")
    logger.save_checkpoint_metrics(f"epoch{epoch}")
```

**Result:** Complete monitoring of training run!

---

## Key Design Decisions

### **1. Why separate profiling from metrics?**

**Profiling** = System-level (GPU, timing)
**Metrics** = Training-level (loss, reward)

Different concerns → separate modules!

---

### **2. Why windowed statistics?**

```python
tracker.get_mean("loss", window=100)  # Last 100 steps
```

**Reason:** Recent trend more important than overall mean!

Example:
- Overall mean loss: 0.5
- Last 100 steps mean: 0.2

Model IS improving, but overall mean doesn't show it!

---

### **3. Why JSON export instead of TensorBoard?**

**JSON advantages:**
- ✅ Simpler (no dependencies)
- ✅ Portable (any tool can read)
- ✅ Version control friendly
- ✅ Can still import to TensorBoard later

**TensorBoard advantages:**
- ✅ Real-time visualization
- ✅ Interactive plots

**Our choice:** JSON first, TensorBoard optional!

---

## Q&A for reference

**Q: How do you detect memory leaks during training?**

**A:**
```python
profiler.set_baseline()  # After model load

for step in range(1000):
    train_step()
    
    if step % 100 == 0:
        if profiler.detect_leak(threshold_gb=0.5):
            print(f"⚠ Leak detected at step {step}")
            # Save state and investigate
```

Baseline = expected memory after model load
Leak = memory grows significantly beyond baseline

---

**Q: What's the overhead of profiling?**

**A:** 
- GPU synchronization: ~0.1ms per measurement
- Minimal memory: Just storing floats
- **Total overhead: <1%** of training time

Worth it for debugging!

---

**Q: How would you optimize training based on profiling?**

**A:**
1. **Find bottleneck** (timing summary)
2. **If forward pass is slow** → Reduce model size / Use mixed precision
3. **If backward pass is slow** → Gradient checkpointing
4. **If data loading is slow** → More workers / Better preprocessing
5. **If memory-bound** → Reduce batch size / Gradient accumulation

Profiling tells you WHERE to optimize!

---

## Key Takeaways

1. **Profile BEFORE training** - Know your limits
2. **Monitor DURING training** - Catch issues early
3. **Export to JSON** - Portable, analyzable
4. **Specialized trackers** - GRPO needs group-wise stats
5. **Windowed statistics** - Recent trends matter most
6. **<1% overhead** - Worth it for reliability