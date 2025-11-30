# Generator Trainer Explained: Complete Guide

## Table of Contents
1. [What is the Generator Trainer?](#what-is-the-generator-trainer)
2. [Self-Verification: The Key Innovation](#self-verification-the-key-innovation)
3. [Reward Function Design](#reward-function-design)
4. [Code Walkthrough](#code-walkthrough)
5. [Training Pipeline](#training-pipeline)
6. [Interview Q&A](#interview-qa)
7. [Common Pitfalls](#common-pitfalls)

---

## What is the Generator Trainer?

The **Generator Trainer** trains a language model to become a proof generator - a model that can:
1. **Generate rigorous proofs** for mathematical problems
2. **Self-verify** its own proofs (identify issues without external help)
3. **Iteratively refine** proofs based on self-identified issues

### Why Self-Verification Matters

**The Problem with Basic Generation:**
```
Generator (without self-verification):
  Input: "Prove √2 is irrational"
  Output: [generates proof]
  
  ❌ No awareness if proof is correct
  ❌ Can't improve on its own
  ❌ Needs external verifier for every attempt
```

**With Self-Verification:**
```
Generator (with self-verification):
  Input: "Prove √2 is irrational"
  Output: [proof] + [self-analysis of the proof]
  
  ✓ Knows when proof has issues
  ✓ Can iteratively refine
  ✓ Scales test-time compute effectively
```

### The Generator's Task

**Output Format:**
```
## Solution
[Mathematical proof]

## Self Evaluation
Here is my evaluation of the solution:
[Analysis: identifies issues, validates steps]

Based on my evaluation, the final overall score should be: \boxed{X}
```

Where X ∈ {0, 0.5, 1} is the generator's self-assessment.

---

## Self-Verification: The Key Innovation

### From Paper (Section 2.2.2)

> "When prompted to both generate and analyze its own proof in one shot, the generator tends to claim correctness even when the external verifier easily identifies flaws."

**The Challenge:**
- Generators have **optimistic bias** - they think their proofs are better than they are
- This creates a **false sense of confidence**
- Iterative refinement fails because generator doesn't see its own mistakes

**The Solution:**
Train the generator to be as critical of its own work as the verifier is.

### How Self-Verification Training Works

**Step 1: Generate proof + self-analysis**
```python
prompt = "Solve this problem. After your solution, evaluate it."
output = generator.generate(prompt)

# Output format:
# ## Solution
# [proof]
# ## Self Evaluation  
# [self-analysis with score]
```

**Step 2: Get two rewards**

**R_Y (Proof Quality):**
```python
proof = extract_proof(output)
r_y = verifier.score(proof)  # 0, 0.5, or 1
```

**R_Z (Self-Evaluation Accuracy):**
```python
self_score = extract_score_from_analysis(output)
r_score = 1 - |self_score - r_y|  # How accurate was self-assessment?

# Meta-verification: Are identified issues real?
r_meta = meta_verifier.check_analysis(proof, self_analysis)

r_z = r_score * r_meta
```

**Step 3: Combined reward**
```python
R = α·R_Y + β·R_Z  # α=0.76, β=0.24 from paper
```

**Step 4: GRPO update**
- Generate K proofs per problem
- Compute combined reward for each
- Use GRPO to prefer high-reward outputs

### Why This Works

**Incentive structure:**
1. **Accurate self-criticism rewarded** - If proof has issues and you identify them → high R_Z
2. **False confidence penalized** - If proof has issues but you claim it's perfect → low R_Z
3. **Best strategy:** Generate good proofs AND honestly evaluate them

**Example:**

**Scenario A: Good proof, honest evaluation**
```
Proof quality (R_Y): 1.0 (perfect)
Self-assessment: "I believe this is correct" → score 1.0
R_score = 1 - |1.0 - 1.0| = 1.0
R_Z = 1.0 * 1.0 = 1.0
R = 0.76×1.0 + 0.24×1.0 = 1.0 ✓
```

**Scenario B: Flawed proof, honest about flaws**
```
Proof quality (R_Y): 0.5 (minor issues)
Self-assessment: "Step 3 needs work" → score 0.5
R_score = 1 - |0.5 - 0.5| = 1.0
R_Z = 1.0 * 1.0 = 1.0
R = 0.76×0.5 + 0.24×1.0 = 0.62
```

**Scenario C: Flawed proof, dishonest (claims perfect)**
```
Proof quality (R_Y): 0.5 (minor issues)
Self-assessment: "This is perfect" → score 1.0
R_score = 1 - |1.0 - 0.5| = 0.5
R_Z = 0.5 * 1.0 = 0.5
R = 0.76×0.5 + 0.24×0.5 = 0.50
```

**Key insight:** Scenario B gets higher reward than C! Honesty pays off.

---

## Reward Function Design

### Component 1: R_format

**Purpose:** Ensure proper output structure

**Checks:**
- Has "## Solution" section
- Has "## Self Evaluation" section  
- Self-analysis contains required phrases
- Has boxed score

**Implementation:**
```python
def check_format(output):
    has_solution = "## Solution" in output
    has_evaluation = "## Self Evaluation" in output
    has_eval_phrase = "Here is my evaluation" in output
    has_score = "\\boxed{" in output
    
    return 1.0 if all([...]) else 0.0
```

### Component 2: R_Y (Proof Quality)

**Purpose:** Reward good proofs

**Computed by:** External verifier

```python
proof = extract_proof_section(output)
r_y = verifier.score(problem, proof)  # Returns 0, 0.5, or 1
```

**This is the main quality signal** - is the proof actually correct?

### Component 3: R_Z (Self-Evaluation Accuracy)

**Purpose:** Reward accurate self-assessment

**Two sub-components:**

**R_score: Score prediction accuracy**
```python
self_predicted = extract_score(self_analysis)  # From generator
verifier_score = r_y  # Ground truth

r_score = 1 - |self_predicted - verifier_score|
```

**R_meta: Analysis faithfulness**
```python
# Use verifier in meta-verification mode
# Check: Are the issues generator identified actually real?
r_meta = meta_verifier.evaluate_analysis(proof, self_analysis)
```

**Combined:**
```python
r_z = r_score * r_meta
```

**Why multiply?** Both must be good:
- Accurate score + hallucinated issues → r_meta=0 → R_Z=0
- Inaccurate score + real issues → r_score low → R_Z low
- Accurate score + real issues → R_Z=1

### Component 4: Total Reward

**Formula from paper (Section 2.2.2):**
```
R = R_format · (α·R_Y + β·R_Z)

where:
  α = 0.76 (proof quality weight)
  β = 0.24 (self-evaluation weight)
```

**Why these weights?**
- **Proof quality (76%)** is most important - generating correct proofs is primary goal
- **Self-evaluation (24%)** is significant - enables refinement and scaling test-time compute
- Still need format, but it's a hard requirement (multiply, not add)

**Example calculations:**

**Perfect everything:**
```
R_format = 1.0
R_Y = 1.0
R_Z = 1.0
R = 1.0 · (0.76×1.0 + 0.24×1.0) = 1.0
```

**Good proof, poor self-eval:**
```
R_format = 1.0
R_Y = 1.0 (proof is perfect)
R_Z = 0.0 (claimed it's flawed)
R = 1.0 · (0.76×1.0 + 0.24×0.0) = 0.76
```

**Poor proof, good self-eval:**
```
R_format = 1.0
R_Y = 0.0 (proof is wrong)
R_Z = 1.0 (correctly identified issues)
R = 1.0 · (0.76×0.0 + 0.24×1.0) = 0.24
```

**Wrong format:**
```
R_format = 0.0
R_Y = 1.0
R_Z = 1.0
R = 0.0 · (anything) = 0.0
```

---

## Code Walkthrough

### 1. GeneratorRewardFunction Class

```python
class GeneratorRewardFunction:
    """
    Reward function for generator with self-verification.
    
    Combines R_Y (proof quality) and R_Z (self-eval accuracy).
    """
    
    def __init__(
        self,
        verifier: Any,
        alpha_proof: float = 0.76,
        beta_self_eval: float = 0.24
    ):
        self.verifier = verifier
        self.alpha_proof = alpha_proof
        self.beta_self_eval = beta_self_eval
```

**Key design:**
- Verifier is a dependency (trained separately)
- Default weights match paper
- Verifier used for both R_Y and R_meta

### 2. Parsing Output

```python
def _parse_output(self, output: str) -> Tuple[str, str]:
    """Split output into proof and self-analysis."""
    
    if "## Solution" in output and "## Self Evaluation" in output:
        parts = output.split("## Self Evaluation")
        proof = parts[0].replace("## Solution", "").strip()
        self_analysis = parts[1].strip()
    else:
        # Fallback
        proof = output
        self_analysis = ""
    
    return proof, self_analysis
```

**Why this matters:**
- Generator outputs both sections in one response
- Need to separate them for individual assessment
- Fallback handles edge cases gracefully

### 3. Computing Rewards

```python
def __call__(self, inputs, outputs):
    """Compute rewards for each output."""
    
    rewards = []
    for inp, output in zip(inputs, outputs):
        # Parse
        proof, self_analysis = self._parse_output(output)
        
        # Check format
        r_format = self._check_format(proof, self_analysis)
        if r_format == 0.0:
            rewards.append(0.0)
            continue
        
        # R_Y: Proof quality
        r_y = self._compute_proof_reward(problem, proof)
        
        # R_Z: Self-evaluation accuracy
        r_z = self._compute_self_eval_reward(
            problem, proof, self_analysis, r_y
        )
        
        # Combine
        reward = r_format * (self.alpha_proof * r_y + self.beta_self_eval * r_z)
        rewards.append(reward)
    
    return rewards
```

**Flow:**
1. Parse output → proof + self-analysis
2. Check format → immediate 0 if invalid
3. Score proof → R_Y from verifier
4. Score self-eval → R_Z from meta-verification
5. Combine → final reward

### 4. GeneratorTrainer Class

```python
class GeneratorTrainer:
    """
    Main trainer for proof generators with self-verification.
    """
    
    def __init__(
        self,
        model: Any,
        verifier: Any,  # Pre-trained verifier
        train_dataset: Any,
        # ... many hyperparameters ...
    ):
        # Initialize reward function
        self.reward_fn = GeneratorRewardFunction(
            verifier=self.verifier,
            alpha_proof=0.76,
            beta_self_eval=0.24
        )
        
        # Initialize GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            reward_fn=self.reward_fn,
            # ... GRPO params ...
        )
```

**Key differences from VerifierTrainer:**
- Requires pre-trained verifier
- Different reward function (R_Y + R_Z vs R_format + R_score)
- Generates longer outputs (proof + analysis)

### 5. Generation with Self-Verification

```python
def generate_proofs_with_self_verification(
    self,
    problems: List[str],
    num_samples: int = 1
) -> Tuple[List[str], Tensor, Tensor]:
    """
    Generate proofs with self-verification.
    
    Returns outputs in format:
    ## Solution
    [proof]
    ## Self Evaluation
    [analysis]
    """
    
    # Would format prompt to request both sections
    # Generate with model
    # Compute log probs
    # Return outputs + log probs
```

**Important:** Single generation produces both proof AND self-analysis.

### 6. Training Step

```python
def train_step(self, batch):
    """One training step."""
    
    problems = batch['problems']
    
    # Generate K proofs with self-verification per problem
    outputs, log_probs, old_log_probs = \
        self.generate_proofs_with_self_verification(
            problems, num_samples=self.group_size
        )
    
    # Prepare for GRPO
    grpo_batch = {
        'inputs': [{'problem': p} for p in problems for _ in range(K)],
        'outputs': outputs,
        'log_probs': log_probs,
        'old_log_probs': old_log_probs
    }
    
    # GRPO step
    stats = self.grpo_trainer.train_step(grpo_batch, self.global_step)
    
    return stats
```

**Flow:**
1. Generate K outputs per problem (each with proof + self-analysis)
2. Compute rewards (R_Y and R_Z for each)
3. GRPO groups by problem
4. Update to prefer high-reward outputs

---

## Training Pipeline

### Data Flow

```
Input: GenerationDataset
├── Item 0: Problem("Prove √2 irrational")
├── Item 1: Problem("Find lim x→∞ (1+1/x)^x")
└── Item 2: Problem("Show ∑1/n² = π²/6")

↓ DataLoader (batch_size=8)

Batch: 8 problems

↓ Generate K=4 outputs per problem

32 outputs, each containing:
  ## Solution
  [proof]
  ## Self Evaluation
  [analysis + score]

↓ Parse each output

32 (proof, self_analysis) pairs

↓ Compute rewards

For each pair:
  - Extract proof → verifier scores → R_Y
  - Extract self-score → compare with R_Y → R_score
  - Meta-verify analysis → R_meta
  - R_Z = R_score * R_meta
  - R = R_format * (0.76·R_Y + 0.24·R_Z)

↓ GRPO: Group by problem (8 groups of 4)

Advantages computed within each group

↓ Policy update

Generator weights updated
```

### Iterative Refinement (Optional)

```
Problem: "Prove √2 is irrational"

Iteration 1:
  Generate: Proof v1 + Self-analysis v1
  Self-score: 0.5 (identified issues in step 3)
  
Iteration 2:
  Prompt: "Here's your proof and analysis. Fix the issues."
  Generate: Proof v2 + Self-analysis v2
  Self-score: 0.5 (still has different issues)
  
Iteration 3:
  Generate: Proof v3 + Self-analysis v3
  Self-score: 1.0 (no more issues found)
  
Stop: Self-score reached 1.0 or max iterations
```

**This is test-time refinement**, not training.

---

## Interview Q&A

### Q1: Why train generator separately from verifier?

**A:**
- **Different objectives:**
  - Verifier: Identify issues (evaluation focus)
  - Generator: Produce proofs (creation focus)
- **Training data:**
  - Verifier: Needs (problem, proof, score) triples
  - Generator: Needs just problems
- **Specialization works better:**
  - Verifier becomes expert critic
  - Generator becomes expert creator
- **Sequential training:**
  - Train verifier first → use as reward model
  - Generator learns from verifier's feedback

### Q2: Why is α=0.76 so much larger than β=0.24?

**A:**
**Primary goal: Generate correct proofs (R_Y)**
- If generator makes perfect proofs but poor self-assessments: still very useful!
- R = 0.76×1.0 + 0.24×0.0 = 0.76 (high reward)

**Secondary goal: Accurate self-verification (R_Z)**
- Enables iterative refinement at test time
- But proof quality matters more

**Analogy:**
- α (76%): "Did you solve the problem correctly?"
- β (24%): "Do you know whether you solved it correctly?"

First question is more important, but second enables self-improvement.

### Q3: How does self-verification help at test time?

**A:**

**Without self-verification:**
```
Generate proof → Submit → Wait for external verification
If wrong: Start over from scratch
```

**With self-verification:**
```
Generate proof + self-analysis
If self-score < 1.0:
  Identify issues from self-analysis
  Refine proof to address issues
  Repeat until self-score = 1.0
Submit final proof
```

**Benefits:**
- **Scales compute:** More iterations → better proofs
- **No external verifier needed** during refinement
- **Targeted fixes:** Knows what to improve

### Q4: What prevents the generator from just lying (claiming proofs are perfect when they're not)?

**A:**

**The R_Z component prevents this:**

**If generator lies:**
```
True quality (R_Y): 0.5
Self-claimed: 1.0
R_score = 1 - |1.0 - 0.5| = 0.5
R_Z = 0.5
R = 0.76×0.5 + 0.24×0.5 = 0.50
```

**If generator is honest:**
```
True quality (R_Y): 0.5
Self-claimed: 0.5
R_score = 1 - |0.5 - 0.5| = 1.0
R_Z = 1.0
R = 0.76×0.5 + 0.24×1.0 = 0.62
```

**Honesty gets 0.62 > lying gets 0.50!**

### Q5: Can the generator improve beyond the verifier?

**A:**

**Initially: No**
- Generator trained with verifier as reward model
- Can't do better than its teacher

**But eventually: Yes, through the synergy cycle (Section 2.3)**
1. Generator improves → produces harder proofs
2. Scale verification compute to label new proofs
3. Retrain verifier on harder examples
4. Use better verifier to train generator further
5. Repeat

Over iterations, both improve together.

### Q6: How many samples (K) should you generate per problem?

**A:**

**Trade-offs:**
- **K=1:** No comparison, high variance, poor learning
- **K=4:** Good balance (used in paper)
- **K=8:** More comparisons, better signal, 2x cost
- **K=16:** Diminishing returns

**Paper uses K=4** for most training.

For **test-time:** Can use larger K (32-64) for important problems.

### Q7: What if self-analysis and verifier disagree?

**A:**

**During training:** Verifier is ground truth
```
Generator says: "Proof is perfect" (score 1.0)
Verifier says: "Has issues" (score 0.5)
R_score = 1 - |1.0 - 0.5| = 0.5  # Penalty for disagreement
```

**At test time:** Trust self-verification
```
No verifier available
Use self-score to decide if refinement needed
This is why accurate self-verification is important!
```

---

## Common Pitfalls

### 1. Not Using Pre-Trained Verifier

```python
# ❌ WRONG: Training generator from scratch
generator_trainer = GeneratorTrainer(
    model=untrained_model,
    verifier=untrained_verifier  # Bad!
)

# ✓ CORRECT: Use pre-trained verifier
verifier = load_pretrained_verifier("checkpoints/verifier/best.pt")
generator_trainer = GeneratorTrainer(
    model=generator_model,
    verifier=verifier  # Good!
)
```

### 2. Wrong Reward Weights

```python
# ❌ WRONG: Equal weights
reward_fn = GeneratorRewardFunction(
    alpha_proof=0.5,
    beta_self_eval=0.5  # Self-eval too high!
)

# ✓ CORRECT: Paper weights
reward_fn = GeneratorRewardFunction(
    alpha_proof=0.76,
    beta_self_eval=0.24
)
```

### 3. Not Parsing Output Correctly

```python
# ❌ WRONG: Treat entire output as proof
proof = output
self_analysis = ""  # Missing!

# ✓ CORRECT: Parse into sections
proof, self_analysis = parse_output(output)
# Now can compute both R_Y and R_Z
```

### 4. Using Generator Self-Score as Ground Truth

```python
# ❌ WRONG: Trust generator's self-assessment
self_score = extract_score(self_analysis)
r_y = self_score  # Bad! Generator can lie

# ✓ CORRECT: Use verifier as ground truth
r_y = verifier.score(proof)  # Verifier is source of truth
r_z = compute_self_eval_accuracy(self_score, r_y)
```

### 5. Infinite Refinement Loops

```python
# ❌ WRONG: Refine forever
while True:
    proof = generate()
    if self_score == 1.0:
        break  # Might never happen!

# ✓ CORRECT: Set max iterations
for iteration in range(max_iterations):
    proof = generate()
    if self_score == 1.0:
        break
```

### 6. Forgetting Format Check

```python
# ❌ WRONG: Compute rewards without format check
r_y = verifier.score(proof)
r_z = compute_self_eval(...)
reward = 0.76*r_y + 0.24*r_z  # Missing R_format!

# ✓ CORRECT: Multiply by format reward
r_format = check_format(output)
reward = r_format * (0.76*r_y + 0.24*r_z)
# If format wrong, reward = 0 regardless of R_Y and R_Z
```

---

## Key Takeaways

1. **Generator produces proof + self-analysis** in single output
2. **Two reward components:** R_Y (proof quality 76%) + R_Z (self-eval accuracy 24%)
3. **Self-verification enables refinement** without external verifier
4. **Honest self-assessment rewarded** - lying about quality is penalized
5. **Uses pre-trained verifier** as ground truth for training
6. **GRPO with K samples** per problem for stable learning
7. **Synergy with verifier** - both improve together iteratively