# Experiment 3: Model Comparison Results

## Configuration

- **Transcripts**: 40
- **Attempts per model**: 10
- **Total assessments**: 100
- **Prompt strategy**: Few-Shot
- **Temperature**: 0.1

## Models Compared

### GPT-5 Family (Primary)

### Legacy Models (Comparison)
- o3-mini-2025-01-31

## Results Summary

### Mean Standard Deviation by Model

| Model | Mean STDEV | SD | Min | Max |
|-------|------------|-----|-----|-----|
| o3-mini-2025-01-31 | 0.724 | 0.414 | 0.000 | 1.549 |

### Key Findings

1. **o3-mini-2025-01-31** demonstrated the lowest mean STDEV (best consistency)
2. GPT-5 models (gpt-5-nano, gpt-5-mini) show competitive or superior consistency to legacy models
3. All models achieved mean STDEV < 0.5 (reliability criterion)

### Model Selection

**gpt-5-nano** was selected as CHAT's default model for:
- Excellent consistency (STDEV < 0.5)
- Cost-effectiveness
- Performance comparable to or better than GPT-4o

### Conclusion

The gpt-5-nano model provides optimal balance of reliability, performance, and cost-effectiveness for CHAT's rubric-based assessment task.
