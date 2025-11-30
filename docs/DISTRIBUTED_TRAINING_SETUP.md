# Distributed Training Infrastructure - Explained

## What We Built (75 words)

**Distributed Training Infrastructure** = Multi-GPU coordination + memory optimization

**distributed_setup.py:**
- Auto-detect single/multi-GPU setup
- DeepSpeed initialization (ZeRO-3)
- Process synchronization

**checkpoint_manager.py:**
- Save/load model checkpoints
- Automatic rotation (keep last N)
- Best checkpoint tracking
- Resume from interruption

**deepspeed_config.json:**
- ZeRO Stage 3 (partition everything)
- CPU offloading (optimizer + params)
- Mixed precision (bfloat16)

Think: Distributed = split model across GPUs to fit in memory.

---

## Why Distributed Training?

### **The Problem:**

**DeepSeek-V3 (37B active params) memory requirements:**

```
Single GPU (16GB T4):
✗ Inference (fp16):  74 GB   → Doesn't fit!
✗ Training (full):  296 GB   → Impossible!
```

**Need:** Somehow train on 16GB GPU

---

### **The Solution: DeepSpeed ZeRO**

**ZeRO = Zero Redundancy Optimizer**

**Normal training (data parallel):**
```
GPU 0: [Full Model] [Full Optimizer] [Full Gradients]
GPU 1: [Full Model] [Full Optimizer] [Full Gradients]  
GPU 2: [Full Model] [Full Optimizer] [Full Gradients]
...

Problem: 8x memory needed for 8 GPUs!
```

**ZeRO-3 (partition everything):**
```
GPU 0: [Model Part 1] [Opt Part 1] [Grad Part 1]
GPU 1: [Model Part 2] [Opt Part 2] [Grad Part 2]
GPU 2: [Model Part 3] [Opt Part 3] [Grad Part 3]
...

Result: 1x memory total, split across GPUs!
```

**Key insight:** Each GPU only holds 1/N of model at a time, fetches other parts as needed.

---

## distributed_setup.py - Deep Dive

### **1. setup_distributed_environment()**

**Auto-detects your setup:**

```python
from src.training.distributed_setup import setup_distributed_environment

config = setup_distributed_environment()

# Single GPU:
# {
#   'distributed': False,
#   'local_rank': 0,
#   'world_size': 1,
#   'device': 'cuda:0'
# }

# Multi-GPU (via torchrun):
# {
#   'distributed': True,
#   'local_rank': 0,    # GPU on THIS machine
#   'rank': 0,          # Global rank across all machines
#   'world_size': 8,    # Total GPUs
#   'device': 'cuda:0'
# }
```

**How it works:**

1. Checks environment variables (`RANK`, `WORLD_SIZE`)
2. If set → Multi-GPU mode
3. If not set but CUDA available → Single GPU
4. Otherwise → CPU

---

### **2. DeepSpeedConfig Class**

**Creates DeepSpeed configuration:**

```python
from src.training.distributed_setup import DeepSpeedConfig

config = DeepSpeedConfig(
    zero_stage=3,                     # ZeRO-3: Partition everything
    gradient_accumulation_steps=4,    # Accumulate over 4 batches
    train_micro_batch_size_per_gpu=1  # 1 sample per GPU per step
)

config.save_config("my_config.json")
```

**What gets configured:**

**Batch sizes:**
```python
train_micro_batch_size_per_gpu = 1      # Per GPU
gradient_accumulation_steps = 4          # Accumulate 4 steps
world_size = 8                           # 8 GPUs

effective_batch_size = 1 * 4 * 8 = 32   # Total batch size
```

**ZeRO Stage 3:**
```python
"zero_optimization": {
  "stage": 3,
  
  # Partition parameters across GPUs
  "offload_param": {
    "device": "cpu",        # Store on CPU when not in use
    "pin_memory": true
  },
  
  # Partition optimizer states across GPUs
  "offload_optimizer": {
    "device": "cpu",        # Store on CPU
    "pin_memory": true
  }
}
```

**Mixed precision (bfloat16):**
```python
"bf16": {
  "enabled": true          # Use bf16 instead of fp32
}
```

---

### **3. Key Helper Functions**

**is_main_process():**
```python
if is_main_process():
    # Only rank 0 does this
    print("Saving checkpoint...")
    save_checkpoint()
```

**Why?** Prevent all GPUs from saving same checkpoint!

---

**get_effective_batch_size():**
```python
effective = get_effective_batch_size(
    train_micro_batch_size_per_gpu=1,
    gradient_accumulation_steps=4,
    world_size=8
)
# Returns: 32
```

**Why?** Know true batch size for learning rate scaling!

---

**synchronize_processes():**
```python
# All GPUs wait here until everyone arrives
synchronize_processes()
print("All GPUs synchronized!")
```

**Why?** Coordinate checkpointing, evaluation, etc.

---

## checkpoint_manager.py - Deep Dive

### **1. CheckpointManager Class**

**Complete checkpoint lifecycle:**

```python
from src.training.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="checkpoints/verifier",
    max_checkpoints=3,        # Keep only last 3
    save_optimizer=True,      # Save optimizer state
    save_scheduler=True       # Save scheduler state
)
```

---

### **2. Saving Checkpoints**

```python
# During training
for step in range(10000):
    loss = train_step()
    
    if step % 1000 == 0:
        manager.save_checkpoint(
            model=model,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'loss': loss, 'reward': 0.8},
            is_best=(loss < best_loss)
        )
```

**What gets saved:**

```
checkpoints/verifier/
├── checkpoint_step_1000/
│   ├── model.pt           # Model weights
│   ├── optimizer.pt       # Optimizer state
│   ├── scheduler.pt       # Scheduler state
│   └── metadata.json      # Step, metrics, timestamp
├── checkpoint_step_2000/
│   └── ...
├── checkpoint_step_3000/
│   └── ...
└── best_checkpoint/       # Copy of best checkpoint
    └── ...
```

---

### **3. Loading Checkpoints**

**Load latest (resume training):**
```python
metadata = manager.load_latest_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler
)

print(f"Resumed from step {metadata['step']}")
# Start training from metadata['step'] + 1
```

**Load best (for evaluation):**
```python
manager.load_best_checkpoint(model=model)
# Model now has weights from best checkpoint
```

**Load specific checkpoint:**
```python
checkpoint_path = Path("checkpoints/verifier/checkpoint_step_5000")
manager.load_checkpoint(checkpoint_path, model)
```

---

### **4. Checkpoint Rotation**

**Problem:** Checkpoints take lots of space (74GB each for DeepSeek-V3!)

**Solution:** Keep only last N checkpoints

```python
manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    max_checkpoints=3   # Keep last 3
)

# Save 5 checkpoints
manager.save_checkpoint(model, step=1000)
manager.save_checkpoint(model, step=2000)
manager.save_checkpoint(model, step=3000)
manager.save_checkpoint(model, step=4000)  # Step 1000 deleted!
manager.save_checkpoint(model, step=5000)  # Step 2000 deleted!

# Only have: 3000, 4000, 5000
```

**Special case:** Best checkpoint NEVER deleted!

---

## deepspeed_config.json - Deep Dive

### **Key Sections:**

**1. Batch Configuration**
```json
{
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4
}
```

**Micro batch** = What fits in GPU memory
**Gradient accumulation** = Simulate larger batch

---

**2. ZeRO Stage 3**
```json
{
  "zero_optimization": {
    "stage": 3,
    
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

**What this does:**
- Parameters live on CPU
- Moved to GPU only when needed for computation
- Optimizer states also on CPU
- Enables training 200B+ models on single GPU!

---

**3. Activation Checkpointing**
```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

**What are activations?** Intermediate values during forward pass, needed for backward pass.

**Problem:** Large models → huge activations (10GB+)

**Solution:** Don't store all activations, recompute during backward pass
- **Tradeoff:** More compute, less memory

---

**4. Mixed Precision**
```json
{
  "bf16": {
    "enabled": true
  }
}
```

**fp32** = 32-bit floats (4 bytes) → 296GB for V3
**bf16** = 16-bit bfloat (2 bytes) → 148GB for V3

**Why bfloat16 > float16?**
- Same exponent range as fp32
- Less likely to overflow/underflow
- Supported on modern GPUs (A100, H100, T4)

---

## How They Work Together

### **Training Setup Flow:**

```python
from src.training.distributed_setup import (
    setup_distributed_environment,
    initialize_deepspeed,
    is_main_process
)
from src.training.checkpoint_manager import CheckpointManager
from src.model.verifier import ProofVerifier

# 1. Setup distributed environment
dist_config = setup_distributed_environment()
print_distributed_config(dist_config)

# 2. Create model
model = ProofVerifier()

# 3. Initialize DeepSpeed
model_engine, optimizer, scheduler, ds_config = initialize_deepspeed(
    model=model,
    config_path="configs/deepspeed_config.json",
    zero_stage=3
)

# 4. Setup checkpoint manager (only on main process)
if is_main_process():
    checkpoint_manager = CheckpointManager("checkpoints/verifier")

# 5. Resume from checkpoint if exists
if is_main_process():
    metadata = checkpoint_manager.load_latest_checkpoint(
        model_engine,
        optimizer,
        scheduler
    )
    start_step = metadata['step'] + 1 if metadata else 0
else:
    start_step = 0

# 6. Training loop
for step in range(start_step, total_steps):
    loss = train_step(model_engine)
    
    # Save checkpoint (only main process)
    if step % 1000 == 0 and is_main_process():
        checkpoint_manager.save_checkpoint(
            model_engine,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'loss': loss}
        )

# 7. Cleanup
cleanup_distributed()
```

---

## Single GPU vs Multi-GPU

### **Single GPU (What we have - T4 16GB):**

```bash
# Run training script normally
python scripts/10_train_verifier.py
```

**What happens:**
- `world_size = 1`
- `distributed = False`
- DeepSpeed still used for ZeRO-3 memory optimization
- CPU offloading enabled
- Can train larger models than normal!

---

### **Multi-GPU (Future - 8x A100):**

```bash
# Run with torchrun
torchrun --nproc_per_node=8 scripts/10_train_verifier.py
```

**What happens:**
- `world_size = 8`
- `distributed = True`
- Each GPU gets rank 0-7
- Model partitioned across all GPUs
- 8x faster training!

---

## Memory Calculation Examples

### **Without ZeRO (Normal Training):**

**7B model, fp16:**
```
Model:     14 GB
Gradients: 14 GB
Optimizer: 28 GB (Adam: 2x fp32 states)
Total:     56 GB → Need A100 (80GB)
```

### **With ZeRO-3 + CPU Offload:**

**Same 7B model:**
```
GPU:
  Activations:       ~8 GB (batch-dependent)
  Working memory:    ~4 GB (temp)
  Total GPU:        ~12 GB → Fits on T4!

CPU:
  Model:            14 GB
  Optimizer:        28 GB
  Total CPU:        42 GB → Normal RAM
```

**Key:** Only activations stay on GPU!

---

## GENERAL Q&A

**Q: What's the difference between ZeRO stages?**

**A:**
- **ZeRO-0:** No partitioning (normal DDP)
- **ZeRO-1:** Partition optimizer states
- **ZeRO-2:** Partition optimizer + gradients
- **ZeRO-3:** Partition optimizer + gradients + model parameters

**Memory savings:**
- Stage 1: ~4x
- Stage 2: ~8x
- Stage 3: ~Nx (N = num GPUs)

---

**Q: Why not just use gradient checkpointing instead of ZeRO?**

**A:** They solve different problems!

**Gradient checkpointing:** Saves memory by not storing activations
- **Saves:** Activation memory
- **Cost:** 30% slower (recompute)

**ZeRO-3:** Partitions model/optimizer/gradients
- **Saves:** Parameter memory
- **Cost:** Communication overhead

**Best:** Use BOTH together!

---

**Q: How does checkpoint rotation work with best checkpoint?**

**A:**
```python
max_checkpoints = 3

Save step 1000 (best)  → [1000-best, 1000]
Save step 2000         → [1000-best, 1000, 2000]
Save step 3000         → [1000-best, 1000, 2000, 3000]
Save step 4000         → [1000-best, 2000, 3000, 4000]  # 1000 NOT deleted!

# Best checkpoint preserved even though it's oldest
```

---

**Q: What happens if training crashes?**

**A:**
```python
# Resume automatically
metadata = checkpoint_manager.load_latest_checkpoint(model, optimizer)

if metadata:
    start_step = metadata['step'] + 1
    print(f"Resuming from step {start_step}")
else:
    start_step = 0
    print("Starting from scratch")

# Continue training
for step in range(start_step, total_steps):
    train_step()
```