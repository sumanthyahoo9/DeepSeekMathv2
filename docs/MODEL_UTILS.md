# model_utils.py - Brief Explanation

## What It Does (50 words)

**model_utils.py** = Swiss Army knife of helper functions for model operations.

**5 categories:**
1. **Generation configs** - Control sampling (temperature, top-p)
2. **Output formatting** - Clean/parse model responses
3. **Score extraction** - Pull scores from \\boxed{} tags
4. **Batch processing** - Split/combine lists
5. **Validation** - Check response completeness

Think: Toolbox that everything else uses.

---

## Key Functions by Category

### **1. Generation Configs**

```python
# Greedy (deterministic, always same output)
config = get_greedy_config(max_new_tokens=2048)
# {'do_sample': False, 'temperature': 1.0, ...}

# Sampling (random, creative)
config = get_sampling_config(temperature=0.7, num_samples=4)
# {'do_sample': True, 'temperature': 0.7, 'num_return_sequences': 4}
```

**What it controls:**
- **temperature** - Higher = more random (0.0 = greedy, 1.0 = very random)
- **top_p** - Nucleus sampling (0.9 = sample from top 90% probable tokens)
- **top_k** - Only sample from top K tokens
- **num_return_sequences** - Generate multiple outputs

**When to use:**
- **Greedy:** Verification (want consistent output)
- **Sampling:** Generation (want diversity)

---

### **2. Output Formatting**

```python
# Remove prompt from generated text
text = "Problem: ...\n\nSolution: This is the proof"
clean = clean_generated_text(text, remove_prompt=True)
# "This is the proof"

# Extract sections
text = "## Solution\nProof...\n## Self Evaluation\nAnalysis..."
parts = extract_solution_and_evaluation(text)
# {'solution': 'Proof...', 'evaluation': 'Analysis...'}

# Truncate at stop tokens
text = "Generated text<|endoftext|>Extra junk"
clean = truncate_at_stop_sequence(text)
# "Generated text"
```

**Why needed:** Models often include prompts in output, need to clean up!

---

### **3. Score Extraction**

```python
# Single score
text = "Analysis... \\boxed{0.5}"
score = extract_score(text)
# 0.5

# Multiple texts
texts = ["\\boxed{1}", "\\boxed{0.5}", "no score"]
scores = extract_multiple_scores(texts)
# [1.0, 0.5, None]
```

**Pattern:** Uses regex to find `\\boxed{...}` and validates score ∈ {0, 0.5, 1}

---

### **4. Batch Processing**

```python
# Split into batches
texts = ['a', 'b', 'c', 'd', 'e']
batches = batch_texts(texts, batch_size=2)
# [['a', 'b'], ['c', 'd'], ['e']]

# Flatten back
flattened = flatten_batches(batches)
# ['a', 'b', 'c', 'd', 'e']
```

**Use case:** Process 1000 samples in batches of 8 for efficiency.

---

### **5. Validation**

```python
# Check required sections exist
response = "## Solution\nProof\n## Self Evaluation\nAnalysis"
valid = validate_response_format(response)
# {'Solution': True, 'Self Evaluation': True}

# Check response not truncated
complete = check_response_completeness(response, min_length=50)
# True (has score or long enough)
```

**Why:** Detect if model output is malformed before using it!

---

## Helper Functions

### **format_parameter_count()**
```python
format_parameter_count(7_000_000_000)  # "7.0B"
format_parameter_count(350_000_000)    # "350.0M"
format_parameter_count(5_000)          # "5.0K"
```
Human-readable model sizes for logging.

---

### **get_model_family()**
```python
get_model_family("deepseek-ai/DeepSeek-V3")  # "deepseek"
get_model_family("meta-llama/Llama-2-7b")    # "llama"
```
Identify model type from name (useful for model-specific logic).

---

### **summarize_batch_results()**
```python
scores = [1.0, None, 0.5, 1.0, None]
summary = summarize_batch_results(scores)
# {
#   'total': 5,
#   'valid': 3,
#   'invalid': 2,
#   'success_rate': 0.6,
#   'mean_score': 0.83,
#   'distribution': {'0.0': 0, '0.5': 1, '1.0': 2}
# }
```
Quick stats for a batch - essential for monitoring training!

---

## Design Patterns

### **Pattern 1: Config Factory Functions**
```python
# Instead of manually creating dicts everywhere:
config = {
    'max_new_tokens': 2048,
    'temperature': 0.7,
    'top_p': 0.9,
    ...  # Easy to forget params!
}

# Use factory:
config = get_sampling_config(temperature=0.7)
# All params set with good defaults
```

**Benefit:** Consistency + fewer bugs from missing params.

---

### **Pattern 2: Pure Functions**
All functions are **pure** (no side effects):
- Input → Output, no global state
- Easy to test
- Easy to parallelize
- Predictable behavior

```python
# Pure function - same input always gives same output
result = extract_score("\\boxed{1}")  # Always returns 1.0

# Not pure (would be bad):
# def extract_score_bad(text):
#     global last_score  # BAD!
#     last_score = ...
```

---

### **Pattern 3: Defensive Validation**
Always check validity:
```python
def extract_score(text: str) -> Optional[float]:
    score = parse_score(text)
    
    # Validate score is valid
    if score not in [0, 0.5, 1]:
        return None  # Don't crash, return None
    
    return score
```

**Benefit:** Graceful degradation instead of crashes.

---

## Key Takeaways

1. **Utilities = Reusable helpers** used everywhere
2. **Config factories** ensure consistency
3. **Pure functions** = testable & predictable
4. **Defensive coding** = handle invalid inputs gracefully
5. **Batch helpers** = efficiency at scale

This module doesn't do anything on its own - it's the toolbox that `verifier.py` and `generator.py` will use!