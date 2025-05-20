# Experiment 4: Few-Shot vs. Zero-Shot Prompting Methods

## Overview

This experiment compares different prompting approaches (few-shot vs. zero-shot) to determine which leads to more accurate and consistent patient communication assessments.

## Approach

1. Design zero-shot prompts (instructions only) and few-shot prompts (instructions with examples)
2. Grade a set of transcripts using both prompting methods
3. Compare the accuracy, consistency, and reliability of each approach
4. Analyze how the number and quality of examples affect grading performance

## Implementation

### Required Files

- High-quality example assessments for few-shot prompting
- Zero-shot prompt templates
- Few-shot prompt templates with varying numbers of examples
- Selected set of transcripts for consistent evaluation

### Scripts

- `01_generate_example_assessments.py`: Create high-quality examples for few-shot prompting
- `02_run_prompting_comparison.py`: Grade transcripts using different prompting methods
- `03_analyze_results.py`: Compare and analyze the performance of each prompting approach
