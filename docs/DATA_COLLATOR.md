# Data Collator Explanation (Simple & Interview-Ready)

## What is a Data Collator? (50 words)

A **collator** takes individual samples from a dataset and combines them into a **batch** for training. It handles:
1. **Text formatting** (combining problem + proof)
2. **Tokenization** (text → numbers)
3. **Padding** (make all sequences same length)
4. **Truncation** (cut if too long)

Think: Assembly line worker preparing ingredient batches for cooking.

---

## Why Do We Need It? (75 words)

**Problem:** Neural networks process batches, not individual samples. Each sample has different text lengths.

**Solution:** Collator makes everything uniform:
- Batch of 8 samples → 8 sequences of SAME length
- Model can process them in parallel on GPU
- Much faster than one-by-one processing

**Analogy:** Like organizing books on a shelf - you need them all the same height (padding) or they fall over. Collator adds "padding" to short texts so batch is rectangular.

---

## The 4 Collator Types

### **1. VerificationCollator**
**Use:** Training proof verifier
**Input:** (problem, proof, score)
**Output:** Combined text + label

```
Input sample:
{
    'problem': 'Prove sqrt(2) is irrational',
    'proof': 'Assume sqrt(2) = p/q...',
    'score': 1.0
}

Formatted text:
"Problem: Prove sqrt(2) is irrational

Proof: Assume sqrt(2) = p/q..."

Label: 1.0 (perfect proof)
```

---

### **2. MetaVerificationCollator**
**Use:** Training meta-verifier (checks verifier quality)
**Input:** (problem, proof, verifier_analysis, meta_score)
**Output:** Combined text + meta_label

```
Input sample:
{
    'problem': 'Test problem',
    'proof': 'Test proof',
    'analysis': 'The proof is correct because...',
    'meta_score': 0.5  # Analysis has minor issues
}

Formatted text:
"Problem: Test problem

Proof: Test proof

Analysis: The proof is correct because..."

Label: 0.5 (decent analysis but not perfect)
```

---

### **3. GenerationCollator**
**Use:** Training proof generator
**Input:** (problem only)
**Output:** Prompt for generation

```
Input sample:
{
    'problem': 'Prove 2+2=4'
}

Formatted text:
"Problem: Prove 2+2=4

Solution:"

No label needed - model generates the proof
```

---

### **4. InferenceCollator**
**Use:** Evaluation/testing (flexible)
**Input:** (problem) OR (problem, proof)
**Output:** Formatted text + preserves metadata

**Special feature:** Keeps ALL original fields (sample_id, category, etc.) for later analysis.

```
Input sample:
{
    'problem': 'Test problem',
    'sample_id': 'test_001',
    'category': 'algebra',
    'difficulty': 'hard'
}

Output:
{
    'texts': ['Problem: Test problem\n\nSolution:'],
    'metadata': [original_sample_with_all_fields]
}
```

---

## Key Concepts Explained

### **Padding**

```
Sample 1: [1, 2, 3]           (length 3)
Sample 2: [4, 5, 6, 7, 8]     (length 5)

After padding to max length (5):
Sample 1: [1, 2, 3, 0, 0]     (added 2 padding tokens)
Sample 2: [4, 5, 6, 7, 8]     (no change)

Now both are length 5 → can batch together!
```

**Two strategies:**
- `padding='longest'` - pad to longest sequence IN THIS BATCH (efficient)
- `padding='max_length'` - pad to absolute max (128K tokens) - wastes memory but consistent

---

### **Truncation**

```
Sample has 150K tokens but max_length=128K

Without truncation: Error! Too long!
With truncation: Cut to first 128K tokens

Trade-off: Lose information vs. fit in memory
```

---

### **Tokenization**

```
Text: "Prove that 2+2=4"
Tokens: [1234, 5678, 2, 3, 2, 4, 9999]
         ↑     ↑    ↑  ↑  ↑  ↑   ↑
       Prove  that  2  +  2  =   4

Model sees numbers, not text!
```

---

## GENERAL Q&A

### Q1: Why not just pad everything to max_length always?
**A:** Memory waste! If max_length=128K but your batch has sequences of length 2K, you're wasting 126K * batch_size of memory. Use `padding='longest'` to only pad to longest in that specific batch.

---

### Q2: What happens if we don't pad?
**A:** Can't create a tensor! PyTorch needs rectangular tensors:
```
# This works:
[[1, 2, 3],
 [4, 5, 6]]  # 2x3 tensor ✓

# This doesn't work:
[[1, 2, 3],
 [4, 5]]     # Ragged! ✗
```

---

### Q3: Why do we need different collators for each phase?
**A:** Different training phases need different inputs:
- **Verifier:** Needs problem + proof → judge quality
- **Generator:** Only needs problem → create proof
- **Meta-verifier:** Needs problem + proof + analysis → judge analysis

Same base logic, different text formatting.

---

### Q4: What's the difference between Dataset and Collator?
**A:**
- **Dataset:** Stores individual samples, retrieves one at a time
- **Collator:** Takes multiple samples, combines into batch

```
Dataset[0] → {'problem': 'P1', 'proof': 'Proof1'}  # Single sample
Dataset[1] → {'problem': 'P2', 'proof': 'Proof2'}  # Single sample

Collator([Dataset[0], Dataset[1]]) → {
    'input_ids': Tensor([[...], [...]]),      # Batched!
    'attention_mask': Tensor([[...], [...]])
}
```

---

### Q5: Why preserve metadata in InferenceCollator?
**A:** After inference, you need to know WHICH sample each prediction came from. 
Metadata links predictions back to original samples:

```
Input metadata: {'sample_id': 'test_001', 'category': 'algebra'}
Model prediction: Score = 0.8

Link them: "Sample test_001 (algebra) got score 0.8"

Needed for analysis, debugging, and reporting results!
```

---

### Q6: How does estimate_batch_memory work?
**A:** Rough calculation:
```
Memory = Model weights + Activations + Overhead

Activations ≈ batch_size × seq_length × hidden_size × num_layers × 2 bytes (fp16)

Example:
- batch_size=4, seq_length=128K, hidden=4096, layers=100
- Activations ≈ 4 × 128K × 4K × 100 × 2 bytes ≈ 400 GB!

This is why batch_size=1 for long sequences!
```

---

## Code Pattern: Factory Function

```python
def get_collator(collator_type: str, **kwargs):
    """
    Factory pattern: Create object based on string input
    
    Instead of:
        if type == 'verification':
            return VerificationCollator(**kwargs)
        elif type == 'meta_verification':
            return MetaVerificationCollator(**kwargs)
        ...
    
    Use dictionary lookup (cleaner):
        collators = {
            'verification': VerificationCollator,
            'meta_verification': MetaVerificationCollator
        }
        return collators[type](**kwargs)
    """
```

**Why?** More maintainable, less if/else spaghetti!

---

## Memory Optimization Tips

1. **Use longest padding** (not max_length) - saves memory
2. **Sort by length** before batching - less padding waste
3. **Gradient accumulation** - simulate larger batch size without memory spike
4. **Mixed precision (fp16)** - half the memory per value

```python
# Bad: All padded to 128K
collator = VerificationCollator(padding='max_length', max_length=128000)

# Good: Only pad to longest in batch
collator = VerificationCollator(padding='longest')

# If batch has sequences [2K, 3K, 5K] → pad to 5K, not 128K!
```

---

## Key Takeaways

1. **Collator = Batch assembler**
2. **Handles padding, truncation, tokenization**
3. **Different collators for different training phases**
4. **InferenceCollator preserves metadata**
5. **Memory estimation helps prevent OOM errors**
6. **Factory pattern for clean object creation**