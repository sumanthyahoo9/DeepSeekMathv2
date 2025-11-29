"""
src/utils/prompts.py

Prompt templates for DeepSeekMath-V2 training and inference.
All prompts are extracted from the paper (Appendix A).

Each function returns a formatted prompt string.
"""

from typing import Optional
import re


# ============================================================================
# Evaluation Rubrics (Used across multiple prompts)
# ============================================================================

VERIFICATION_RUBRICS = """Please evaluate the solution and score it according to the following criteria:
- If the solution is completely correct, with all steps executed properly and clearly demonstrated, then the score is 1
- If the solution is generally correct, but with some details omitted or minor errors, then the score is 0.5
- If the solution does not actually address the required problem, contains fatal errors, or has severe omissions, then the score is 0
- Additionally, referencing anything from any paper does not save the need to prove the reference. It's okay IF AND ONLY IF the solution also presents a valid proof of the reference argument(s); otherwise, if the solution omits the proof or if the proof provided is not completely correct, the solution should be scored according to the criteria above, and definitely not with a score of 1"""


# ============================================================================
# Proof Generation Prompts
# ============================================================================

def get_proof_generation_prompt(problem: str) -> str:
    """
    Prompt for generating proofs with self-verification.
    
    Args:
        problem: The mathematical problem to solve
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""Your task is to solve a given problem. The problem may ask you to prove a statement, or ask for an answer. If finding an answer is required, you should come up with the answer, and your final solution should also be a rigorous proof of that answer being valid.

Your final solution to the problem should be exceptionally comprehensive and easy-to-follow, which will be rated according to the following evaluation instruction:

'''
{VERIFICATION_RUBRICS}
'''

In fact, you already have the ability to rate your solution yourself, so you are expected to reason carefully about how to solve a given problem, evaluate your method according to the instruction, and refine your solution by fixing issues identified until you can make no further progress.

In your final response, you should present a detailed solution to the problem followed by your evaluation of that solution.

- To give a good final response, you should try your best to locate potential issues in your own (partial) solution according to the evaluation instruction above, and fix them as many as you can.
- A good final response should just faithfully present your progress, including the best solution you can give, as well as a faithful evaluation of that solution.
- Only when you fail to locate any issues in your solution should you score it with 1.
- If you do notice some issues in your solution but fail to resolve them with your best efforts, it's totally ok to faithfully present the issues in your final response.
- The worst final response would provide a wrong solution but lie that it's correct or claim that it's correct without careful error checking. A better version should faithfully identify errors in the solution. Remember! You CAN'T cheat! If you cheat, we will know, and you will be penalized!

Your final response should be in the following format:

## Solution
[Your final solution to the problem here. You should try your best to optimize the quality of your solution according to the evaluation instruction above before finalizing it here.]

## Self Evaluation
Here is my evaluation of the solution:
[Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution. You should analyze your solution faithfully. E.g., if there are issues in your final solution, you should point it out.]

Based on my evaluation, the final overall score should be: \\boxed{{[0, 0.5, or 1]}}

---

Here is your task input:

## Problem
{problem}"""
    
    return prompt


# ============================================================================
# Proof Verification Prompts
# ============================================================================

def get_proof_verification_prompt(problem: str, proof: str) -> str:
    """
    Prompt for verifying a proof and assigning a score.
    
    Args:
        problem: The mathematical problem
        proof: The proposed solution/proof
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""## Instruction
Your task is to evaluate the quality of a solution to a problem. The problem may ask for a proof of statement, or ask for an answer. If finding an answer is required, the solution should present the answer, and it should also be a rigorous proof of that answer being valid.

{VERIFICATION_RUBRICS}

Please carefully reason out and analyze the quality of the solution below, and in your final response present a detailed evaluation of the solution's quality followed by your score. Therefore, your response should be in the following format:

Here is my evaluation of the solution:
[Your evaluation here. You are required to present in detail the key steps of the solution or the steps for which you had doubts regarding their correctness, and explicitly analyze whether each step is accurate: for correct steps, explain why you initially doubted their correctness and why they are indeed correct; for erroneous steps, explain the reason for the error and the impact of that error on the solution.]

Based on my evaluation, the final overall score should be: \\boxed{{[0, 0.5, or 1]}}

---

Here is your task input:

## Problem
{problem}

## Solution
{proof}"""
    
    return prompt


# ============================================================================
# Meta-Verification Prompts
# ============================================================================

def get_meta_verification_prompt(problem: str, proof: str, proof_analysis: str) -> str:
    """
    Prompt for meta-verifying a verifier's analysis.
    
    Args:
        problem: The mathematical problem
        proof: The proposed solution
        proof_analysis: The verifier's analysis of the proof
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are given a "problem", "solution", and "solution evaluation", and you need to assess whether this "solution evaluation" is reasonable.

First, "solution evaluation" is generated to evaluate the quality of the "solution", by prompting a verifier with the rules below (these are not your rules):

'''
{VERIFICATION_RUBRICS}
'''

Next, I will introduce the rules for you to analyze the quality of the "solution evaluation":

1. Your task is to analyze the "solution evaluation". You do not need to solve the "problem", nor do you need to strictly assess whether the "solution" is accurate. Your only task is to strictly follow the rules below to evaluate whether the "solution evaluation" is reasonable.

2. You need to analyze the content of the "solution evaluation" from three aspects:

**Step Restatement**: In the "solution evaluation", certain behaviors of the "solution" may be restated. You need to return to the original text of the "solution" and check whether the "solution" actually has these behaviors mentioned in the "solution evaluation".

**Defect Analysis**: "solution evaluation" may point out errors or defects in the "solution". You need to carefully analyze whether the mentioned errors and defects are indeed valid.

**Expression Analysis**: Whether the "solution evaluation"'s expressions are accurate.

**Score Analysis**: Whether the final score given by the "solution evaluation" matches the defects it found. You need to analyze according to the scoring rules given above.

3. The most important part is **defect analysis**: In this part, your core task is to check whether the errors or defects of the "solution" pointed out in the "solution evaluation" are reasonable. In other words, any positive components about the "solution" in the "solution evaluation", regardless of whether they are reasonable, are not within your evaluation scope.

- For example: If the "solution evaluation" says that a certain conclusion in the "solution" is correct, but actually this conclusion is incorrect, then you do not need to care about this point. All parts that the "solution evaluation" considers correct do not belong to your evaluation scope.
- Specifically: If the "solution evaluation" believes that the "solution" is completely accurate and has not found any errors or defects, then regardless of whether the "solution" itself is actually accurate, even if there are obvious errors, you should still consider its analysis of errors to be reasonable.

**Importantly**, for defects found by the "solution evaluation", you need to analyze two points simultaneously:
- whether this defect actually exists
- whether the "solution evaluation"'s analysis of this defect is accurate

These two aspects constitute the analysis of defects.

4. About **expression analysis**, if there are certain expression errors in the "solution evaluation", even minor errors in details, you need to identify them. However, please note that identifying incorrect steps in the "solution" as correct steps does not constitute an **expression error**. In practice, expression errors include but are not limited to:

- If the "solution evaluation" identifies some reasoning step(s) in the "solution" as incorrect, then it cannot further indicate that subsequent conclusion(s) depending on those reasoning step(s) are wrong, but can only indicate that subsequent conclusion(s) are "not rigorously demonstrated."
- Typos and calculation errors made by "solution evaluation"
- Inaccurate restatement of content from "solution"

5. Finally, you need to present your analysis of the "solution evaluation" in your output and also rate its quality based on the rules below:

First, if there is at least one unreasonable defect among the defects found by the "solution evaluation", then you only need to do **defect analysis**:
- If all defects found by the "solution evaluation" are unreasonable, then you should rate it with 0
- If some defects found by the "solution evaluation" are reasonable and some are unreasonable, then your rating should be 0.5

Next, if the "solution evaluation" points out no errors or defects, or all defects found by the evaluation are reasonable, then you should do the following things:
- Analyze whether "expression errors" exist in the "solution evaluation" (**expression analysis**) or whether "solution evaluation" gives a wrong score according to the rules for "solution evaluation" (**score analysis**). If yes, you should rate the "solution evaluation" with 0.5; if no, your rating should be 1

Your output should follow the format below:

Here is my analysis of the "solution evaluation":
[Your analysis here.]

Based on my analysis, I will rate the "solution evaluation" as: \\boxed{{[0, 0.5, or 1]}}

---

Here is your task input:

## Problem
{problem}

## Solution
{proof}

## Solution Evaluation
{proof_analysis}"""
    
    return prompt


# ============================================================================
# Proof Refinement Prompts
# ============================================================================

def get_proof_refinement_prompt(
    problem: str,
    previous_proof: str,
    proof_analyses: str
) -> str:
    """
    Prompt for refining a proof based on verification feedback.
    
    Args:
        problem: The mathematical problem
        previous_proof: The previous proof attempt
        proof_analyses: One or more analyses of the previous proof
        
    Returns:
        Formatted prompt string
    """
    # Get the base generation prompt
    base_prompt = get_proof_generation_prompt(problem)
    
    # Add refinement context
    refinement_section = f"""

## Candidate Solution(s) to Refine
Here are some solution sample(s) along with their correctness evaluation(s). You should provide a better solution by solving issues mentioned in the evaluation(s), or by re-using promising ideas mentioned in the solution sample(s), or by doing both.

{previous_proof}

{proof_analyses}

## Final Instruction
Your final response should follow the format above, including a '## Solution' section followed by a '## Self Evaluation' section."""
    
    # Insert refinement section before the problem
    prompt = base_prompt + refinement_section
    
    return prompt


# ============================================================================
# Utility Functions for Parsing Responses
# ============================================================================

def extract_score_from_response(response: str) -> Optional[float]:
    """
    Extract score from model response.
    
    Args:
        response: The model's response text
        
    Returns:
        Score (0, 0.5, or 1) or None if not found
    """
    
    # Look for \boxed{score} pattern
    pattern = r'\\boxed\{([0-9.]+)\}'
    matches = re.findall(pattern, response)
    
    if matches:
        try:
            score = float(matches[-1])  # Take the last match
            if score in [0, 0.5, 1, 1.0]:
                return score if score != 1.0 else 1
        except ValueError:
            pass
    
    return None


def extract_sections_from_response(response: str) -> dict:
    """
    Extract solution and self-evaluation sections from response.
    
    Args:
        response: The model's response text
        
    Returns:
        Dictionary with 'solution' and 'evaluation' keys
    """
    sections = {}
    
    # Split by markdown headers
    parts = response.split('## ')
    
    for part in parts:
        part = part.strip()
        if part.lower().startswith('solution'):
            # Remove 'Solution' header and extract content
            content = part[len('solution'):].strip()
            # Stop at next ## header
            if '## ' in content:
                content = content.split('## ')[0].strip()
            sections['solution'] = content
            
        elif part.lower().startswith('self evaluation'):
            content = part[len('self evaluation'):].strip()
            sections['evaluation'] = content
    
    return sections


def check_format_compliance(response: str, response_type: str = "generation") -> bool:
    """
    Check if response follows the required format.
    
    Args:
        response: The model's response text
        response_type: Type of response ("generation", "verification", or "meta_verification")
        
    Returns:
        True if format is correct, False otherwise
    """
    if response_type == "generation":
        # Should have both ## Solution and ## Self Evaluation sections
        has_solution = '## Solution' in response
        has_evaluation = '## Self Evaluation' in response
        has_eval_phrase = 'Here is my evaluation of the solution:' in response
        has_score_phrase = 'Based on my evaluation, the final overall score should be:' in response
        has_boxed = '\\boxed{' in response
        
        return all([has_solution, has_evaluation, has_eval_phrase, has_score_phrase, has_boxed])
        
    elif response_type == "verification":
        # Should have evaluation phrase and score
        has_eval_phrase = 'Here is my evaluation of the solution:' in response
        has_score_phrase = 'Based on my evaluation, the final overall score should be:' in response
        has_boxed = '\\boxed{' in response
        
        return all([has_eval_phrase, has_score_phrase, has_boxed])
        
    elif response_type == "meta_verification":
        # Should have analysis phrase and score
        has_analysis_phrase = 'Here is my analysis of the "solution evaluation":' in response
        has_score_phrase = 'Based on my analysis, I will rate the "solution evaluation" as:' in response
        has_boxed = '\\boxed{' in response
        
        return all([has_analysis_phrase, has_score_phrase, has_boxed])
    
    return False


# ============================================================================
# Export all functions
# ============================================================================

__all__ = [
    'VERIFICATION_RUBRICS',
    'get_proof_generation_prompt',
    'get_proof_verification_prompt',
    'get_meta_verification_prompt',
    'get_proof_refinement_prompt',
    'extract_score_from_response',
    'extract_sections_from_response',
    'check_format_compliance',
]