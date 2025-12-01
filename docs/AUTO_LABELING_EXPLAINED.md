# Auto-Labeling Pipeline Explained: Complete Guide

## Table of Contents
1. [What is Auto-Labeling?](#what-is-auto-labeling)
2. [The Problem It Solves](#the-problem-it-solves)
3. [The Algorithm Step-by-Step](#the-algorithm-step-by-step)
4. [Code Walkthrough](#code-walkthrough)
5. [Usage Examples](#usage-examples)
6. [Interview Q&A](#interview-qa)
7. [Common Pitfalls](#common-pitfalls)

---

## What is Auto-Labeling?

**Auto-labeling** is the process of automatically assigning correctness scores to proofs **without human annotation**, using scaled verification and consensus mechanisms.

### From Paper (Section 2.3)

> "Building on these observations, we developed the following automated labeling process..."

The paper's key insight: **Scaling verification compute can replace human experts** for labeling new training data.

### The Core Idea

**Traditional labeling:**
```
New proof → Human expert → Score (0, 0.5, or 1)
Problem: Slow, expensive, doesn't scale
```

**Auto-labeling:**
```
New proof → n verifier analyses → m meta-checks → Consensus → Score
Advantage: Fast, cheap, scales with compute
```

---

## The Problem It Solves

### The Training Plateau

**Scenario:**
```
1. Train verifier on dataset D_v (17,503 problems)
2. Use verifier to train generator
3. Generator improves → creates harder proofs
4. Verifier struggles with harder proofs
5. ❌ Can't train generator further without better verifier
```

**Need:** More labeled data to improve verifier, but:
- Manual annotation is slow (experts needed)
- Harder proofs take longer to verify
- Can't scale to thousands of new proofs

### The Synergy Cycle

**Solution: Auto-label generator's outputs**
```
Iteration 1:
  Verifier v1 → trains Generator g1
  
Iteration 2:
  Generator g1 → creates new proofs
  Auto-label new proofs → Dataset D_v2
  Train Verifier v2 on D_v + D_v2
  
Iteration 3:
  Verifier v2 → trains Generator g2
  Generator g2 → creates harder proofs
  Auto-label → Dataset D_v3
  Train Verifier v3 on D_v + D_v2 + D_v3
  
[Continue cycle...]
```

**Result:** Both verifier and generator improve together!

---

## The Algorithm Step-by-Step

### Paper Parameters (Section 2.3)

- **n = 64** verification analyses per proof
- **m = 32** meta-verification checks per analysis  
- **k = 8** threshold for consensus

### Full Algorithm

```
For each proof to label:

  STEP 1: Generate n verification analyses
    analyses = []
    for i in range(64):
      analysis = verifier.analyze(proof)
      analyses.append(analysis)
  
  STEP 2: Filter analyses that found issues
    issue_analyses = [a for a in analyses if a.score < 1.0]
    
    If no issues found → label = 1.0, done!
  
  STEP 3: Meta-verify each issue analysis
    valid_analyses = []
    for analysis in issue_analyses:
      meta_checks = []
      for j in range(32):
        meta_check = meta_verifier.check(analysis)
        meta_checks.append(meta_check)
      
      # Majority vote
      if sum(meta_checks) / 32 >= 0.5:
        valid_analyses.append(analysis)
  
  STEP 4: Determine label via consensus
    if len(valid_analyses) >= 8:
      # At least k=8 analyses found real issues
      label = min(a.score for a in valid_analyses)
    else:
      # No consensus on issues
      label = 1.0
```

### Concrete Example

**Proof:** "Prove √2 is irrational"

**Step 1: Generate 64 analyses**
```
Analysis 1: "Proof is correct" → score=1.0
Analysis 2: "Step 3 unclear" → score=0.5
Analysis 3: "Missing justification" → score=0.5
...
Analysis 64: "Proof is correct" → score=1.0

Result: 45 say score=1.0, 19 say score=0.5
```

**Step 2: Filter issues (score < 1.0)**
```
19 analyses found issues
```

**Step 3: Meta-verify each of the 19**
```
Analysis 2 ("Step 3 unclear"):
  Run 32 meta-checks
  Results: 12 say "valid issue", 20 say "not a real issue"
  12/32 = 0.375 < 0.5 → NOT valid
  
Analysis 3 ("Missing justification"):
  Run 32 meta-checks
  Results: 25 say "valid issue", 7 say "not real"
  25/32 = 0.78 > 0.5 → VALID
  
... [continue for all 19]

Result: 6 analyses have valid issues
```

**Step 4: Consensus**
```
Valid analyses: 6 < threshold (k=8)
No consensus that proof has issues
→ Label = 1.0 (assume correct)
```

### Why This Works

**Key observations from paper:**

1. **Scaling samples increases issue detection**
   - 1 sample might miss issues
   - 64 samples much more likely to catch problems

2. **Meta-verification prevents hallucinations**
   - Verifier might claim fake issues
   - Meta-verifier filters false positives

3. **k-threshold ensures robustness**
   - Need multiple independent confirmations
   - Reduces noise from random errors

---

## Code Walkthrough

### 1. ScaledVerification Class

```python
class ScaledVerification:
    """
    Automatically label proofs using scaled verification.
    
    Parameters from paper (Section 2.3):
    - n_verifications: 64
    - m_meta_checks: 32
    - k_threshold: 8
    """
    
    def __init__(
        self,
        verifier: Any,
        meta_verifier: Optional[Any] = None,
        n_verifications: int = 64,
        m_meta_checks: int = 32,
        k_threshold: int = 8,
        meta_consensus_threshold: float = 0.5
    ):
        self.verifier = verifier
        self.meta_verifier = meta_verifier or verifier  # Can use same model
        self.n_verifications = n_verifications
        self.m_meta_checks = m_meta_checks
        self.k_threshold = k_threshold
```

**Design choice:** Meta-verifier can be same model as verifier, just with different prompting (meta-verification mode).

### 2. Generate Verification Analyses

```python
def generate_verification_analyses(
    self,
    problem: str,
    proof: str,
    proof_id: str
) -> List[VerificationAnalysis]:
    """Generate n independent analyses of the proof."""
    
    analyses = []
    
    # Generate with sampling (temperature > 0)
    for i in range(self.n_verifications):
        analysis = self.verifier.generate(
            prompt=verification_prompt(problem, proof),
            temperature=0.8  # Diversity important!
        )
        
        # Parse score and issues
        score = extract_score(analysis)
        has_issues = (score < 1.0)
        
        analyses.append(VerificationAnalysis(
            proof_id=f"{proof_id}_analysis_{i}",
            analysis_text=analysis,
            score=score,
            has_issues=has_issues
        ))
    
    return analyses
```

**Why temperature=0.8?** Need diversity to catch different types of issues. Temperature=0 would always produce same analysis.

### 3. Meta-Verification

```python
def meta_verify_analysis(
    self,
    problem: str,
    proof: str,
    analysis: VerificationAnalysis
) -> List[MetaVerificationResult]:
    """Check if the analysis is faithful (issues are real)."""
    
    results = []
    
    for i in range(self.m_meta_checks):
        meta_analysis = self.meta_verifier.generate(
            prompt=meta_verification_prompt(problem, proof, analysis.text),
            temperature=0.8
        )
        
        # Extract: are the issues real?
        is_valid = check_issues_valid(meta_analysis)
        quality = extract_quality_score(meta_analysis)
        
        results.append(MetaVerificationResult(
            analysis_id=f"{analysis.proof_id}_meta_{i}",
            is_valid=is_valid,
            quality_score=quality,
            meta_analysis=meta_analysis
        ))
    
    return results
```

### 4. Consensus Computation

```python
def compute_meta_consensus(
    self,
    meta_results: List[MetaVerificationResult]
) -> Tuple[bool, float]:
    """Majority vote across meta-checks."""
    
    num_valid = sum(1 for r in meta_results if r.is_valid)
    total = len(meta_results)
    
    # Majority vote
    is_valid = (num_valid / total) >= self.meta_consensus_threshold
    confidence = num_valid / total if is_valid else (total - num_valid) / total
    
    return is_valid, confidence
```

**Example:**
```
32 meta-checks: 25 say "valid", 7 say "not valid"
25/32 = 0.78 >= 0.5 → is_valid = True
confidence = 0.78
```

### 5. Label Assignment

```python
def label_proof(
    self,
    problem: str,
    proof: str,
    proof_id: str
) -> AutoLabelResult:
    """Main auto-labeling function."""
    
    # Step 1: Generate n analyses
    analyses = self.generate_verification_analyses(problem, proof, proof_id)
    
    # Step 2: Filter issue analyses
    issue_analyses = [a for a in analyses if a.has_issues]
    
    # Step 3: Meta-verify each
    valid_analyses = []
    for analysis in issue_analyses:
        meta_results = self.meta_verify_analysis(problem, proof, analysis)
        is_valid, confidence = self.compute_meta_consensus(meta_results)
        
        if is_valid:
            valid_analyses.append(analysis)
    
    # Step 4: Consensus labeling
    if len(valid_analyses) >= self.k_threshold:
        label = min(a.score for a in valid_analyses)
        reasoning = f"{len(valid_analyses)} valid analyses found issues"
    else:
        label = 1.0
        reasoning = f"Only {len(valid_analyses)} < {self.k_threshold} valid analyses"
    
    return AutoLabelResult(
        proof_id=proof_id,
        label=label,
        confidence=...,
        reasoning=reasoning
    )
```

---

## Usage Examples

### Example 1: Auto-label existing proofs

```bash
python scripts/20_auto_label.py \
    --verifier checkpoints/verifier/best.pt \
    --proofs data/unlabeled_proofs.jsonl \
    --output data/auto_labeled.jsonl \
    --n_verifications 64 \
    --m_meta_checks 32 \
    --k_threshold 8
```

**Output:**
```
data/auto_labeled.jsonl:
{"proof_id": "proof_0", "problem": "...", "proof": "...", "score": 1.0, "confidence": 0.95, ...}
{"proof_id": "proof_1", "problem": "...", "proof": "...", "score": 0.5, "confidence": 0.78, ...}
...
```

### Example 2: Generate and auto-label

```bash
python scripts/20_auto_label.py \
    --generator checkpoints/generator/best.pt \
    --verifier checkpoints/verifier/best.pt \
    --problems data/problems.jsonl \
    --output data/generated_labeled.jsonl \
    --generate \
    --num_proofs_per_problem 4
```

**Process:**
1. Loads problems
2. Generates 4 proofs per problem
3. Auto-labels all generated proofs
4. Saves labeled dataset

### Example 3: Quick test with small parameters

```bash
python scripts/20_auto_label.py \
    --verifier checkpoints/verifier/best.pt \
    --proofs data/test_proofs.jsonl \
    --output data/test_labeled.jsonl \
    --n_verifications 8 \
    --m_meta_checks 4 \
    --k_threshold 2 \
    --debug
```

Faster for testing (but less accurate than paper parameters).

---

## GENERAL Q&A

### Q1: Why do we need n=64 analyses? Isn't that expensive?

**A:** 
**Trade-off:** Accuracy vs. cost

**Single analysis (n=1):**
- Might miss subtle issues
- High false positive/negative rate
- Unreliable labels

**64 analyses:**
- Much more likely to find real issues
- Statistical consensus is robust
- Cost: ~64x more compute

**Paper's insight:** In later training iterations, human annotation becomes the bottleneck. Spending compute on auto-labeling is worth it.

**Also:** This is offline labeling. Do once, use for many epochs.

### Q2: Can the verifier verify its own outputs (self-meta-verification)?

**A:** Yes! The paper uses the **same model** for both:
- Verification mode: "Analyze this proof"
- Meta-verification mode: "Check if this analysis is accurate"

Different prompting, same weights.

**Why this works:** Easier to critique an analysis than to create one from scratch.

### Q3: What if consensus is never reached (valid_analyses always < k)?

**A:** Label as correct (score=1.0) with low confidence.

**Reasoning:**
- No clear evidence of issues
- Multiple verification attempts failed to find problems
- Conservative: assume correct unless proven otherwise

**Alternative:** Discard proof (don't label). Paper mentions this as option.

### Q4: How does k-threshold prevent noise?

**A:**

**Without k-threshold (k=1):**
```
1 random analysis finds fake issue
→ Label = 0.5 (wrong!)
```

**With k=8:**
```
1 random analysis finds fake issue
But only 1 valid analysis < 8
→ Label = 1.0 (correct!)
```

**k acts as noise filter** - need multiple independent confirmations.

### Q5: What's the confidence score used for?

**A:**

**High confidence (0.9+):**
- Strong consensus
- Use for training immediately

**Medium confidence (0.6-0.9):**
- Some disagreement
- Still useful for training

**Low confidence (<0.6):**
- High disagreement
- Consider discarding or human review

**Can filter dataset** by confidence before retraining verifier.

### Q6: How does this enable the synergy cycle?

**A:**

**The cycle:**
```
1. Train Verifier v1 on human-labeled D_v
2. Use v1 to train Generator g1
3. Generator g1 creates new proofs (harder)
4. Auto-label new proofs → D_v2
5. Train Verifier v2 on D_v + D_v2 (better!)
6. Use v2 to train Generator g2
7. Repeat...
```

**Key:** Auto-labeling removes human bottleneck. Can generate and label thousands of proofs per iteration.

### Q7: Does this replace human annotation entirely?

**A:** In paper: Yes for last 2 iterations!

**Quote from Section 2.3:**
> "In our last two training iterations, this fully automated pipeline replaced human annotation entirely."

**Process:**
1. **Iteration 1:** Human annotate initial dataset
2. **Iterations 2-3:** Mix of human + auto-labeling
3. **Iterations 4+:** Pure auto-labeling

**Quality check:** "Quality checks confirmed that the automated labels aligned well with expert judgments."

---

## Common Pitfalls

### 1. Too Few Verification Samples

```python
# ❌ WRONG: n=4 (not enough)
scaled_verifier = ScaledVerification(
    verifier=verifier,
    n_verifications=4,  # Too small!
    k_threshold=8  # But threshold still 8
)
# Problem: Can never reach k=8 with only 4 samples!
```

**Rule:** `n_verifications >= k_threshold * 2` at minimum

### 2. k-threshold Too High

```python
# ❌ WRONG: k too close to n
scaled_verifier = ScaledVerification(
    n_verifications=64,
    k_threshold=60  # Too high!
)
# Problem: Almost impossible to reach consensus
# Everything labeled as 1.0
```

**Paper ratio:** k=8, n=64 → k is 12.5% of n

### 3. Forgetting Temperature > 0

```python
# ❌ WRONG: Temperature = 0 (deterministic)
for i in range(64):
    analysis = verifier.generate(temp=0.0)  # Always same!

# ✓ CORRECT: Temperature > 0 for diversity
for i in range(64):
    analysis = verifier.generate(temp=0.8)  # Different each time
```

### 4. Not Using Meta-Verification

```python
# ❌ WRONG: Skip meta-verification
valid_analyses = issue_analyses  # Trust all!

# ✓ CORRECT: Meta-verify to filter false positives
valid_analyses = []
for analysis in issue_analyses:
    if meta_verify(analysis):
        valid_analyses.append(analysis)
```

### 5. Wrong Consensus Logic

```python
# ❌ WRONG: Average scores
label = mean([a.score for a in valid_analyses])

# ✓ CORRECT: Minimum score (most conservative)
label = min([a.score for a in valid_analyses])
```

**Reasoning:** If even one valid analysis says score=0, there's likely a serious issue.

### 6. Not Saving Confidence Scores

```python
# ❌ WRONG: Only save labels
save_dataset(problems, proofs, labels)

# ✓ CORRECT: Save labels + confidence + reasoning
save_dataset(problems, proofs, labels, confidences, reasoning)
# Can filter low-confidence samples later
```

---

## Key Takeaways

1. **Auto-labeling = Scaled verification + Consensus**
2. **Paper parameters:** n=64, m=32, k=8
3. **Enables synergy:** Verifier and generator improve together
4. **Replaced human annotation** in later training iterations
5. **Meta-verification filters false positives**
6. **k-threshold ensures robustness** against noise
7. **Temperature > 0 for diversity** in analyses
8. **Confidence scores** help filter dataset quality