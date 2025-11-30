# Model Wrappers - Verifier & Generator Explained

## What They Do (75 words)

**verifier.py** and **generator.py** wrap the base LLM for specific tasks:

**ProofVerifier:**
- Input: (problem, proof)
- Output: Analysis + Score (0, 0.5, 1)
- Purpose: Judge proof quality

**MetaVerifier:**
- Input: (problem, proof, verifier_analysis)
- Output: Meta-analysis + Meta-score
- Purpose: Check if verifier hallucinated issues

**ProofGenerator:**
- Input: (problem)
- Output: Proof + Self-evaluation + Self-score
- Purpose: Create proofs with self-checking

Think: Specialized adapters for the same base model.

---

## Architecture Overview

```
┌─────────────────────────────────────┐
│      Base LLM (DeepSeek-V3)        │
│         (shared weights)            │
└─────────────────────────────────────┘
         │           │           │
         ▼           ▼           ▼
   ┌─────────┐ ┌──────────┐ ┌──────────┐
   │Verifier │ │  Meta-   │ │Generator │
   │  π_φ    │ │ Verifier │ │   π_θ    │
   │         │ │   π_η    │ │          │
   └─────────┘ └──────────┘ └──────────┘
      Score      Validate     Generate
     0/0.5/1     Verifier      Proof
```

**Key insight:** Same base model, different prompts + fine-tuning!

---

## ProofVerifier Deep Dive

### **Basic Usage**

```python
from src.model.verifier import ProofVerifier

verifier = ProofVerifier()

problem = "Prove that sqrt(2) is irrational"
proof = "Assume sqrt(2) = p/q in lowest terms..."

result = verifier.verify(problem, proof)

print(result['analysis'])  # "The proof correctly applies..."
print(result['score'])     # 1.0
```

### **What happens inside `verify()`?**

**Step 1: Create prompt**
```python
prompt = get_proof_verification_prompt(problem, proof)
# Adds instructions: "Analyze this proof, identify issues, assign score"
```

**Step 2: Generate with model**
```python
# Use greedy decoding (deterministic)
outputs = model.generate(inputs, do_sample=False)
```

**Step 3: Parse output**
```python
analysis = clean_generated_text(output)
score = extract_score(analysis)  # Looks for \boxed{score}
```

**Step 4: Return structured result**
```python
return {
    'analysis': "The proof is...",
    'score': 0.5,
    'raw_output': "..."
}
```

---

### **Batch Verification**

```python
problems = ["P1", "P2", "P3"]
proofs = ["Proof1", "Proof2", "Proof3"]

results = verifier.verify_batch(problems, proofs)
# Returns list of 3 results
```

**Why batch?** More efficient than 3 separate `verify()` calls!

---

### **Mock Mode (No GPU)**

When no model loaded, uses length heuristic:
- **Short proof (<50 chars)** → score = 0.0
- **Medium (50-200 chars)** → score = 0.5  
- **Long (200+ chars)** → score = 1.0

**Purpose:** Can develop/test without downloading 236B model!

---

## MetaVerifier Deep Dive

### **What is Meta-Verification?**

**Problem:** Verifier might hallucinate issues that don't exist.

**Solution:** Meta-verifier checks: "Is the verifier's analysis actually valid?"

### **Usage**

```python
from src.model.verifier import MetaVerifier

meta_verifier = MetaVerifier()

# First, get verifier's analysis
verifier_result = verifier.verify(problem, proof)

# Then, meta-verify it
meta_result = meta_verifier.meta_verify(
    problem,
    proof,
    verifier_result['analysis']
)

print(meta_result['meta_score'])  # 1.0 = good analysis, 0.0 = hallucinated
```

### **Training Pipeline**

```
Iteration 1:
  Verifier → No meta-verifier feedback
  
Iteration 2:
  Verifier → Gets meta-verifier feedback
  Result: Fewer hallucinated issues!
```

---

## ProofGenerator Deep Dive

### **Basic Generation**

```python
from src.model.generator import ProofGenerator

generator = ProofGenerator(enable_self_verification=True)

problem = "Prove that sqrt(2) is irrational"

result = generator.generate(problem)

print(result['solution'])     # The actual proof
print(result['evaluation'])   # Self-assessment
print(result['self_score'])   # 0.0, 0.5, or 1.0
```

### **What happens inside `generate()`?**

**Step 1: Create prompt**
```python
prompt = get_proof_generation_prompt(problem)
# Includes: "Generate proof AND self-evaluate it"
```

**Step 2: Generate with sampling** (not greedy!)
```python
# Use temperature=0.7 for creativity
outputs = model.generate(inputs, temperature=0.7, do_sample=True)
```

**Step 3: Parse output**
```python
parts = extract_solution_and_evaluation(output)
# Splits on "## Solution" and "## Self Evaluation" headers

return {
    'solution': parts['solution'],
    'evaluation': parts['evaluation'],
    'self_score': extract_score(parts['evaluation'])
}
```

---

### **Parallel Search** (Multiple Candidates)

```python
# Generate 4 different proofs
results = generator.generate_multiple(
    problem,
    num_samples=4,
    temperature=0.8  # Higher = more diverse
)

# Pick best one based on self-scores
best = max(results, key=lambda r: r['self_score'] or 0)
```

**Why?** Increases chance of finding correct proof!

---

### **Sequential Refinement** (Iterative Improvement)

```python
problem = "Prove that..."

# Step 1: Generate initial proof
initial = generator.generate(problem)

# Step 2: Verify it
verification = verifier.verify(problem, initial['solution'])

# Step 3: If not perfect, refine
if verification['score'] < 1.0:
    refined = generator.refine(
        problem,
        previous_proof=initial['solution'],
        feedback=verification['analysis']
    )
    # refined['solution'] should be better!
```

**Strategy from paper:**
- Try up to 3 refinements
- Stop if score = 1.0
- Use refined proof as final answer

---

## Key Design Decisions

### **1. Why separate classes for Verifier/Generator?**

**Could do:**
```python
model = BaseProofModel()
model.verify(...)
model.generate(...)
```

**Instead do:**
```python
verifier = ProofVerifier()
generator = ProofGenerator()
```

**Why?**
- Different default configs (greedy vs sampling)
- Different post-processing logic
- Clearer intent
- Can load different checkpoints

---

### **2. Why use greedy for verification?**

**Greedy** = deterministic, always same output
**Sampling** = random, different every time

**Verification needs consistency:**
- Same proof should get same score
- Reduces variance in training
- More reliable for auto-labeling

**Generation needs diversity:**
- Multiple attempts at solving
- Explore different approaches

---

### **3. Why mock mode?**

**Real model:** 236B params, needs multi-GPU
**Our GPU:** 1x T4, 16GB → Can't fit!

**Mock mode lets us:**
- ✅ Develop all code on CPU
- ✅ Test logic without model
- ✅ Build training pipeline
- ✅ Write documentation
- Later: Swap in smaller model (7B) or use LoRA

---

## GENERAL Q&A

**Q: Why not just use one model for everything?**

**A:** We DO use one base model! But we fine-tune 3 copies:
- Verifier specializes in scoring proofs
- Meta-verifier specializes in checking verifiers
- Generator specializes in creating proofs

Specialization improves performance on each task.

---

**Q: How does self-verification training work?**

**A:** Model learns to:
1. Generate proof
2. Identify its own mistakes
3. Get reward for BOTH good proof AND accurate self-assessment

Reward formula: `α·R_Y + β·R_Z` where:
- R_Y = Proof quality (from external verifier)
- R_Z = Self-assessment accuracy
- α=0.76, β=0.24 (emphasize proof quality)

This incentivizes honest self-evaluation!

---

**Q: What if verifier gives wrong score?**

**A:** That's what meta-verifier prevents!

Training loop:
1. Verifier trained on expert labels
2. Meta-verifier checks for hallucinations
3. Verifier retrained with meta-feedback
4. Result: Fewer false positives

---

**Q: Why return dictionaries instead of objects?**

**A:** 
```python
# Could do:
class VerificationResult:
    analysis: str
    score: float

# Instead:
return {'analysis': ..., 'score': ...}
```

**Why dictionaries:**
- Easier serialization (JSON)
- Flexible (can add fields)
- Less boilerplate
- Pythonic for data pipelines

---

**Q: How would you add a new verification metric?**

**A:**
```python
# In verifier.py
def verify(self, problem, proof):
    result = {
        'analysis': ...,
        'score': ...,
        'confidence': self._compute_confidence(...)  # NEW!
    }
    return result
```

Dictionary structure makes this easy!

---

## Common Patterns

### **Pattern 1: Verification Pipeline**

```python
verifier = ProofVerifier()
meta_verifier = MetaVerifier()

# Standard verification
v_result = verifier.verify(problem, proof)

# Double-check with meta-verifier
if v_result['score'] < 0.5:
    # Low score - verify verifier isn't hallucinating
    m_result = meta_verifier.meta_verify(
        problem, proof, v_result['analysis']
    )
    
    if m_result['meta_score'] < 0.5:
        # Verifier hallucinated, ignore low score
        print("False negative detected!")
```

---

### **Pattern 2: Best-of-N Sampling**

```python
generator = ProofGenerator()
verifier = ProofVerifier()

# Generate N candidates
candidates = generator.generate_multiple(problem, num_samples=8)

# Verify all
scores = []
for candidate in candidates:
    v = verifier.verify(problem, candidate['solution'])
    scores.append(v['score'])

# Pick best
best_idx = scores.index(max(scores))
best_proof = candidates[best_idx]
```

---

### **Pattern 3: Iterative Refinement**

```python
MAX_ITERATIONS = 3

current_proof = generator.generate(problem)

for iteration in range(MAX_ITERATIONS):
    # Verify current proof
    v = verifier.verify(problem, current_proof['solution'])
    
    if v['score'] >= 1.0:
        break  # Perfect! Stop here
    
    # Refine based on feedback
    current_proof = generator.refine(
        problem,
        current_proof['solution'],
        v['analysis']
    )

final_proof = current_proof['solution']
```

---

## Key Takeaways

1. **Same base model, different fine-tuning** for verifier/generator
2. **Mock mode** enables CPU development without GPU
3. **Greedy for verification** (consistency), **sampling for generation** (diversity)
4. **Meta-verifier** prevents hallucinated issues
5. **Self-verification** incentivizes honest self-assessment
6. **Dictionary returns** for flexibility
7. **Modular design** allows easy experimentation