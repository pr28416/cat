# Experiment 2: Real Data Assessment

## Overview

This experiment evaluates the performance of LLMs on real-world patient-doctor transcripts from the DoVA dataset.

## Approach

1. Grade real transcripts using both non-rubric and rubric-based approaches
2. Analyze the results to assess LLM consistency and reliability
3. Compare results against baseline synthetic transcript assessments

## Implementation

### Required Files

- Cleaned DoVA transcripts in `data/cleaned/`
- Non-rubric grading prompt
- Rubric-based grading prompt
- Patient Health Communication Rubric v5.0

### Scripts

- `01_run_real_transcript_grading.py`: Grade real transcripts using LLMs
- `02_analyze_results.py`: Analyze grading results and generate insights
