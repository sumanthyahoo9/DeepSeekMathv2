# base_model.py - Brief Explanation

## What It Does (50 words)

**BaseProofModel** is a wrapper for loading/saving/managing pre-trained LLMs. It handles:
1. **Loading** models from HuggingFace
2. **Saving/loading checkpoints** during training
3. **Device management** (CPU/GPU placement)
4. **Memory estimation** (how much RAM/VRAM needed)
5. **Model info** (parameter count, size)

Think: Universal remote control for any LLM.

---

## Key Methods

### **1. `load_model()`**
```python
model = BaseProofModel("deepseek-ai/DeepSeek-V3.2-Exp-SFT")
model.load_model()  # Downloads from HuggingFace
```
**What it does:** Downloads model from HuggingFace, loads into memory, configures device placement.

---

### **2. `save_checkpoint(path)` / `load_checkpoint(path)`**
```python
# During training
model.save_checkpoint("./checkpoints/epoch_1")

# Resume later
model2 = BaseProofModel()
model2.load_checkpoint("./checkpoints/epoch_1")
```
**What it does:** Saves model weights to disk / loads them back. Essential for resuming interrupted training.

---

### **3. `get_model_size()`**
```python
info = model.get_model_size()
# {'total_params': 7_000_000_000, 'size_gb': 13.5, ...}
```
**What it does:** Counts parameters, estimates memory usage. Helps verify model loaded correctly.

---

### **4. `to(device)`**
```python
model.to("cuda:0")  # Move to GPU 0
model.to("cpu")     # Move to CPU
```
**What it does:** Moves model between devices. Rarely needed (auto-handled during load), but useful for debugging.

---

## Key Utility Functions

### **`estimate_model_memory()`**
```python
memory_gb = estimate_model_memory(
    num_parameters=7_000_000_000,  # 7B model
    dtype="float16",
    include_gradients=True,   # Training needs gradients
    include_optimizer=True    # Adam stores 2 states/param
)
# Returns: ~60 GB needed
```

**Formula:**
- **Weights:** params × bytes_per_dtype
- **Gradients:** params × bytes_per_dtype (for training)
- **Optimizer (Adam):** params × 4 bytes × 2 states

**Why useful?** Check if model fits in GPU RAM BEFORE loading it!

---

### **`count_parameters()`**
```python
params = count_parameters(model)
# {'total_params': 7B, 'trainable_params': 7B, 'frozen_params': 0}
```
**What it does:** Counts total, trainable, and frozen parameters.

---

## Design Choices Explained

### **Why wrap the model?**
- **Consistency:** Same interface for all models (verifier, generator)
- **Utilities:** Built-in checkpoint saving, memory estimation
- **Mock mode:** Can test code without downloading 50GB models

---

### **Why support int8/fp16/bf16?**
Different precision = different memory/speed trade-offs:

| Precision | Bytes/param | Memory (7B) | Speed | Quality |
|-----------|-------------|-------------|-------|---------|
| float32 | 4 | ~28 GB | Slow | Best |
| float16 | 2 | ~14 GB | Fast | Good |
| bfloat16 | 2 | ~14 GB | Fast | Good |
| int8 | 1 | ~7 GB | Fastest | Okay |

**DeepSeek default:** bfloat16 (good balance)

---

### **Why `device="auto"`?**
Automatically places model on best available device:
- If GPU available → use GPU
- If multiple GPUs → distribute across them
- If no GPU → use CPU (slow but works)

---

## GENERAL Q&A

**Q: What's the difference between saving a checkpoint vs. the full model?**

**A:** 
- **Checkpoint:** Just weights (state_dict). Smaller, needs model code to load.
- **Full model:** Weights + architecture. Larger, standalone.

For training, we save checkpoints. For deployment, we save full models.

---

**Q: Why estimate memory before loading?**

**A:** GPU has limited RAM (e.g., T4 = 16GB). If model needs 20GB, loading fails. Better to check first!

---

**Q: What's `device_map="auto"`?**

**A:** HuggingFace feature that automatically distributes model across available GPUs/CPU. Handles big models that don't fit on single GPU.

---

**Q: Why include gradients/optimizer in memory estimate?**

**A:**
- **Inference:** Only need model weights
- **Training:** Need weights + gradients + optimizer states
- Training uses 3-4x more memory than inference!

---

## Key Takeaways

1. **BaseProofModel = Universal LLM wrapper**
2. **Handles loading, saving, device management**
3. **Mock mode allows testing without actual models**
4. **Memory estimation prevents OOM errors**
5. **Supports different precisions (fp32/fp16/int8)**

This forms the foundation for verifier.py and generator.py!