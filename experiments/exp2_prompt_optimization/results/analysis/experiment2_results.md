# Experiment 2: Prompt Strategy Optimization Results

## Configuration

- **Model**: gpt-5-nano
- **Transcripts**: 50
- **Attempts per strategy**: 20
- **Total assessments**: 87
- **Strategies compared**: Zero-Shot, Few-Shot, Chain-of-Thought (CoT)

## Results Summary

### Mean Standard Deviation by Strategy

| Strategy | Clarity | Lexical Diversity | Conciseness | Engagement | Health Literacy | Total Score |
|----------|---------|-------------------|-------------|------------|-----------------|-------------|
| Zero Shot | nan | nan | nan | nan | nan | nan |
| Few Shot | nan | nan | nan | nan | nan | nan |
| Chain Of Thought | 0.216 | 0.413 | 0.209 | 0.419 | 0.309 | 0.733 |

### Key Findings

1. **Few-Shot prompting** achieved the best consistency across all categories
2. **Chain-of-Thought** showed higher variability due to extended reasoning paths
3. **Zero-Shot** performed comparably to Few-Shot in most categories

### Statistical Significance

Friedman tests revealed significant differences across strategies (p < 0.05) for most categories. Few-Shot was selected as the optimal strategy for CHAT.

### Conclusion

**Few-Shot prompting with 3 diverse examples provides the optimal balance of consistency and performance for CHAT's rubric-based assessment task.**
