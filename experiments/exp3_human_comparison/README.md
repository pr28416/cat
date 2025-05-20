# Experiment 3: Human Evaluator Comparison

## Overview

This experiment compares LLM-based assessments against human evaluator assessments to validate the quality and alignment of LLM-generated scores.

## Approach

1. Compare human evaluator scores with LLM-generated scores for both real and synthetic transcripts
2. Calculate correlation coefficients and agreement metrics
3. Analyze areas of disagreement to identify potential model limitations
4. Evaluate inter-rater reliability among human evaluators and compare to LLM consistency

## Implementation

### Required Files

- Human evaluator scores dataset
- LLM-graded transcript scores from previous experiments
- Rubric v5.0 for reference

### Scripts

- `01_import_human_evaluations.py`: Import and format human evaluation data
- `02_compare_human_llm_scores.py`: Calculate correlation and agreement metrics
- `03_analyze_disagreements.py`: Analyze qualitative patterns in disagreements
