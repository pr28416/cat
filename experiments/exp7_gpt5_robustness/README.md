# Experiment 7: GPT-5 Series Robustness Check

## Rationale

With the release of GPT-5, GPT-5.1, and GPT-5.2, we need to assess whether CHAT's reliability generalizes across these newer model generations. This experiment focuses specifically on **reliability robustness**, NOT validation.

## Research Question

**Does CHAT maintain high reliability (STDEV < 0.5) across newer GPT-5 series models?**

## Study Design

### Focused Reliability Assessment

Unlike Experiment 5 (comprehensive validation), this is a **targeted reliability check**:

- **Sample**: 50 transcripts (stratified sample from Exp5)
- **Models**: GPT-5 (latest), GPT-5.1 (if available), GPT-5.2 (if available)
- **Attempts per transcript**: 10 (sufficient for STDEV estimation)
- **Total grading attempts**: 50 × 10 × N_models = 500-1500

### Sampling Strategy

Use stratified sampling from Exp5 transcripts:

- **By data source**: 50% DoVA, 50% Nature
- **By score range**: Balanced across low/medium/high CHAT scores
- **By consistency**: Include mix of high and moderate consistency transcripts from Exp5

### Analysis Plan

1. **Primary Outcome: Reliability (STDEV)**

   - Compute per-transcript STDEV for each model
   - Compare to Exp5 GPT-4o baseline
   - Report: mean, median, 95th percentile
   - **Success criterion**: Mean STDEV < 0.5 (same as Exp5)

2. **Secondary Outcomes**

   - Score distributions (vs Exp5 baseline)
   - Categorical agreement with Exp5 scores
   - Systematic shifts in scores

3. **Statistical Comparisons**
   - Mann-Whitney U: Compare STDEV distributions
   - Spearman's ρ: Correlate mean scores with Exp5
   - **Goal**: Demonstrate similar or improved reliability

### What This Experiment Does NOT Claim

❌ Does NOT re-validate CHAT (validation rests on Exp5)
❌ Does NOT assess human agreement for GPT-5
❌ Does NOT replace the comprehensive Exp5 analysis

✅ ONLY demonstrates reliability robustness to model updates

## Implementation

### Configuration

```python
MODELS_TO_TEST = [
    "gpt-5",          # Latest GPT-5 (if available)
    "gpt-5.1",        # GPT-5.1 (if available)
    "gpt-5.2"         # GPT-5.2 (if available)
]

SAMPLE_SIZE = 50
ATTEMPTS_PER_TRANSCRIPT = 10
TEMPERATURE = 0.3  # Match Exp5 settings
```

### Runtime Estimate

- **Time per grading**: ~10-15 seconds
- **Total API calls**: 50 × 10 × N_models = 500-1500
- **Runtime**: ~2-4 hours (with rate limiting)
- **Cost**: Approximately $50-150 (depending on GPT-5 pricing)

## Expected Outcomes

### Best Case: Reliability Maintained or Improved

- GPT-5 mean STDEV ≤ GPT-4o mean STDEV
- Strong correlation (ρ > 0.90) with Exp5 scores
- **Claim**: "CHAT reliability generalizes to newer model generations"

### Realistic Case: Slight Degradation

- GPT-5 mean STDEV slightly higher but still < 0.5
- Strong correlation (ρ > 0.85)
- **Claim**: "CHAT maintains acceptable reliability across GPT-5 series, with minor variability"

### Challenge Case: Significant Degradation

- GPT-5 mean STDEV > 0.5
- **Interpretation**: Newer models may require prompt refinement or additional few-shot examples
- **Still publishable**: Report honestly, discuss as area for future optimization

## Integration with Main Paper

### Methods Section

- Add brief subsection: "Model Generation Robustness Check (GPT-5)"
- 1-2 paragraphs describing focused reliability assessment

### Results Section

- Report GPT-5 reliability metrics in comparison table
- Include visualization (boxplot of STDEV across models)
- ~1 paragraph

### Discussion

- Frame as evidence of generalizability (if successful)
- Or frame as need for ongoing calibration (if challenges arise)

## Files in This Directory

- `00_select_robustness_sample.py` - Select 50-transcript sample from Exp5
- `01_run_gpt5_grading.py` - Execute GPT-5 grading with checkpointing
- `02_analyze_gpt5_robustness.py` - Compare reliability metrics to Exp5
- `results/` - Output directory for scores and analysis

## Status

**Current Status**: Design complete, ready for implementation when GPT-5 API available

**Dependencies**:

- OpenAI GPT-5 API access
- Exp5 baseline results (completed)

**Note**: If GPT-5 is not yet available, this experiment can be deferred to a post-publication follow-up study or preprint update.
