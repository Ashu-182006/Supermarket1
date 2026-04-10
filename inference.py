# Hackathon Submission Fix Guide

## Issue Summary
Your Phase 2 submission is failing at **Task Validation** with the error:
> "Not enough tasks with graders · One or more task scores are out of range"

This is a **dual error** that indicates:
1. The graders are properly configured (you have 3 tasks) ✅
2. BUT task scores are being clamped incorrectly ⚠️

## Root Cause Analysis

### The Problem
The error message "scores out of range" means the submission evaluator is receiving task scores that are **exactly 0.0 or exactly 1.0**, which violates the constraint: **"Each task's score must be strictly between 0 and 1 (not 0.0 and not 1.0)"**

This can happen if:
- Episodes complete with 0.0 raw reward (complete failure)
- Episodes complete with perfect reward equal to max_reward (perfect success)
- The `clamp` parameter in graders processes data incorrectly
- The `normalised_score()` method returns boundary values in edge cases

### Current Implementation Review

Your `supermart_env.py` has this implementation (line 248-251):
```python
def normalised_score(self) -> float:
    raw = max(0.0, self._total_reward)
    score = round(min(raw / self._max_reward, 1.0), 6) if self._max_reward > 0 else 0.0
    return max(0.001, min(0.999, score))  # ✅ Already clamped!
```

Your `inference.py` also clamps (line 200-202):
```python
raw_score = state.get("normalised_score", 0.0)
return max(0.001, min(0.999, raw_score))  # ✅ Double safety clamp!
```

Your `openenv.yaml` has proper clamp in graders:
```yaml
clamp: [0.001, 0.999]  # ✅ Correct!
```

## The Real Issue: Edge Cases in Reward Calculation

The problem is likely in how `_total_reward` is calculated and clamped. Looking at the code flow:

1. **Perfect episodes**: If an agent completes perfectly, `_total_reward` might equal `_max_reward` exactly → score = 1.0 → **OUT OF RANGE** ❌
2. **Failed episodes**: If an agent gets 0 reward, score = 0.0 → **OUT OF RANGE** ❌

### Solution: Ensure Scoring Never Hits Boundaries

The issue is in the normalised_score calculation. Even with clamping, if your raw score calculation produces exactly 0.0 or 1.0, the clamping converts it to 0.001 or 0.999 AFTER it's returned, but the grader might be seeing the intermediate values.

## Recommended Fix

### Fix 1: Update `openenv.yaml` (Cleaner Approach)

The current configuration is correct, but let's make it more explicit:

```yaml
evaluation:
  tasks:
    - id: task_easy
      reset_body:
        level: easy
        seed: 42
      grader:
        type: score
        field: normalised_score
        aggregation: mean
        episodes: 10
        clamp: [0.001, 0.999]  # ✅ Keep this - ensures scores stay in range

    - id: task_medium
      reset_body:
        level: medium
        seed: 42
      grader:
        type: score
        field: normalised_score
        aggregation: mean
        episodes: 10
        clamp: [0.001, 0.999]

    - id: task_hard
      reset_body:
        level: hard
        seed: 42
      grader:
        type: score
        field: normalised_score
        aggregation: mean
        episodes: 10
        clamp: [0.001, 0.999]
```

### Fix 2: Update `supermart_env.py` (More Defensive)

Strengthen the normalised_score method to ensure it never returns exact 0.0 or 1.0:

```python
def normalised_score(self) -> float:
    raw = max(0.0, self._total_reward)
    if self._max_reward > 0:
        score = min(raw / self._max_reward, 1.0)
    else:
        score = 0.0
    
    # Add small epsilon to avoid exact boundaries
    score = max(0.0, min(1.0, score))
    
    # Ensure strictly between 0 and 1
    epsilon = 1e-3
    if score <= 0.0:
        return epsilon  # 0.001
    elif score >= 1.0:
        return 1.0 - epsilon  # 0.999
    else:
        return score  # Already in range
```

### Fix 3: Verify `inference.py` Return Value

Ensure the final score returned is ALWAYS in range:

```python
# At line 200-202, already correct:
raw_score = state.get("normalised_score", 0.0)
final_score = max(0.001, min(0.999, raw_score))
```

This is already correct ✅

## Implementation Steps

### Step 1: Replace `openenv.yaml`
Copy `openenv_fixed.yaml` to `openenv.yaml` (they're identical in structure, confirming your config is correct)

### Step 2: Update `supermart_env.py` (Recommended)

Replace the `normalised_score` method (around line 248) with:

```python
def normalised_score(self) -> float:
    """Calculate normalised score, ensuring it's strictly between 0 and 1."""
    raw = max(0.0, self._total_reward)
    
    # Calculate base score
    if self._max_reward > 0:
        score = min(raw / self._max_reward, 1.0)
    else:
        score = 0.0
    
    # Clamp strictly between 0.001 and 0.999
    epsilon = 0.001
    return max(epsilon, min(1.0 - epsilon, score))
```

### Step 3: Test Locally

Before resubmitting:
```bash
# Run a quick test with one episode
python inference.py --level easy --base-url http://localhost:7860
```

Verify the final score logged is between 0.001 and 0.999 (not 0.0 or 1.0).

### Step 4: Resubmit

Push your changes and resubmit to the hackathon platform.

## Verification Checklist

- [ ] `openenv.yaml` has exactly 3 tasks (easy, medium, hard) ✅
- [ ] Each task has a grader with `clamp: [0.001, 0.999]` ✅
- [ ] `normalised_score()` method in `supermart_env.py` clamps output
- [ ] `inference.py` returns final score in range [0.001, 0.999]
- [ ] All graders use `type: score` and `field: normalised_score` ✅
- [ ] No scores are exactly 0.0 or 1.0

## Why This Error Happens

The OpenEnv evaluator is strict about score ranges to prevent:
- Division by zero in metrics
- Numerical instability in aggregation
- Invalid probability distributions (scores must support averaging)

The clamp `[0.001, 0.999]` leaves room for averaging without hitting boundaries.

## Need Help?

If you're still seeing failures after these fixes:
1. Check the detailed error logs from the platform
2. Ensure your `normalised_score()` is using the updated code
3. Verify episodes are actually completing (not timing out)
4. Check that `_max_reward` is calculated correctly for each difficulty level
