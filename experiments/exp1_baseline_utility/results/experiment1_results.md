# Experiment 1: Baseline Utility Assessment

## Overview

This experiment evaluated the utility of our LLM-based health communication assessment system by generating synthetic patient-doctor transcripts with known target scores and comparing them against LLM-generated assessments using both non-rubric and rubric-based approaches.

## Comprehensive Description

### Purpose

The primary objective of Experiment 1 was to establish a baseline understanding of how well our LLM-based system can assess health communication quality. This was achieved by:

1. Creating a controlled dataset of synthetic transcripts with predetermined quality scores
2. Evaluating the system's ability to accurately assess these transcripts by comparing LLM-generated scores against the synthetic target scores
3. Comparing the consistency (intra-transcript score variation) and accuracy (Mean Absolute Error against target scores) of rubric-based versus non-rubric assessment approaches

### Experimental Design

The experiment was structured in three main phases:

1. **Synthetic Transcript Generation**

   - Generated 50 unique patient-doctor dialogue transcripts
   - Each transcript was designed to target a specific score (ranging from 5-20)
   - Transcripts were created to represent various communication scenarios and quality levels
   - Used `GPT-4.1` with temperature=0.7 for synthetic transcript generation

2. **Assessment Implementation**

   - Each of the 50 transcripts was assessed 100 times: 50 times using a non-rubric approach (G1) and 50 times using a rubric-based approach (G2)
   - Non-rubric (G1): Direct scoring without explicit criteria
   - Rubric-based (G2): Scoring using the "Patient Health Communication Rubric v5.0"
   - Assessments were performed using `gpt-4.1-mini` with temperature=0.3

3. **Analysis and Evaluation**
   - Collected and processed 5000 total grading attempts (2500 for G1, 2500 for G2)
   - Calculated Mean Absolute Error (MAE) of LLM scores against the `TargetTotalScore_Synthetic` for each transcript and method
   - Calculated the standard deviation (STDEV) of the 50 scores for each transcript under each method (G1 and G2) to assess consistency
   - Conducted statistical tests (Mann-Whitney U for STDEV, Wilcoxon signed-rank for MAE) to compare G1 and G2
   - No parsing errors or missing scores were encountered during processing

### Technical Implementation

- **Synthetic Transcript Generation Model**: `GPT-4.1` (temperature=0.7)
- **Assessment Model**: `gpt-4.1-mini` (temperature=0.3)
- **Assessment Methods**:
  - Non-rubric (G1): Direct scoring based on model's understanding
  - Rubric-based (G2): Structured scoring using "Patient Health Communication Rubric v5.0"
- **Data Processing**:
  - Automated transcript generation and assessment
  - Systematic collection and analysis of results from 5000 grading attempts
  - Statistical analysis of scoring patterns, consistency (STDEV of scores per transcript), and accuracy (MAE against synthetic targets)

### Key Metrics

- Total synthetic transcripts: 50
- Grading attempts per transcript: 100 (50 for non-rubric, 50 for rubric-based)
- Total assessments: 5000
- Target scoring range for synthetic transcripts: 5-20
- Assessment methods: 2 (Non-rubric G1, Rubric-based G2)

### Data Collection

- Generated synthetic transcripts (including `TargetTotalScore_Synthetic`) were loaded from `data/synthetic/exp1_synthetic_transcripts.csv` (not explicitly mentioned in logs, assumed)
- Raw assessment results were stored in:
  - `experiments/exp1_baseline_utility/results/exp1_non_rubric_grading_results.csv`
  - `experiments/exp1_baseline_utility/results/exp1_rubric_grading_results.csv`
- Processed results (all 5000 attempts) were combined in:
  - `experiments/exp1_baseline_utility/results/processed_scores/exp1_all_grading_attempts.csv`
- Summary statistics from the analysis were saved to:
  - `experiments/exp1_baseline_utility/results/analysis/exp1_summary_statistics.csv`
- Generated plots were saved to `experiments/exp1_baseline_utility/results/analysis/` including:
  - `exp1_stdev_distribution.png`
  - `exp1_mae_distribution.png`
  - `exp1_mean_scores_vs_tts_distribution.png`

### Quality Control

- Implemented checkpointing for transcript generation
- Automated error handling and validation; 0 rows dropped due to parsing errors or missing scores
- Consistent model parameters for generation and for assessment
- Systematic data collection and processing

## Methodology

- Generated 50 synthetic transcripts with predefined `TargetTotalScore_Synthetic` values (ranging 5-20)
- Each transcript was graded 50 times using a non-rubric approach (G1) and 50 times using a rubric-based approach (G2)
- All assessments were performed using `gpt-4.1-mini` (temperature=0.3)
- Consistency was measured by the STDEV of the 50 scores for each transcript per method
- Accuracy was measured by the MAE between the mean LLM score (per transcript/method) and the `TargetTotalScore_Synthetic`

## Results

### Statistical Analysis

The primary hypotheses focused on consistency (H1a: Rubric-based grading G2 will have lower STDEV) and accuracy (H1b: Rubric-based grading G2 will have lower MAE against target scores).

| Metric                     | Group1 (Non-Rubric) Median (IQR) | Group2 (Rubric-Based) Median (IQR) | Statistic (Test)         | P-Value | Significance | Hypothesis Supported |
| -------------------------- | -------------------------------- | ---------------------------------- | ------------------------ | ------- | ------------ | -------------------- |
| **H1a: STDEV Total Score** | 0.000 (0.000)                    | 0.000 (0.400)                      | U = 882.0 (Mann-Whitney) | 0.00119 | p < 0.01     | No                   |
| **H1b: MAE vs TTS**        | 4.000 (6.595)                    | 4.980 (8.050)                      | W = 206.5 (Wilcoxon)     | 0.00207 | p < 0.01     | No                   |

_TTS: TargetTotalScore_Synthetic_

**Note on Score Distribution (Mean, Median, Overall STDEV, Range):**
The previous version of this document included a table with overall descriptive statistics for G1 and G2 scores (Mean, Median, Std Dev, Range). While the provided console output focuses on STDEV (consistency per transcript) and MAE (accuracy vs. TTS), the `exp1_summary_statistics.csv` and `exp1_mean_scores_vs_tts_distribution.png` likely contain this broader score distribution information. This section should be updated once that data is reviewed. For now, previous findings on score distribution (e.g., ceiling effects) are retained but may need revision.

### Key Findings

1. **Grading Consistency (STDEV of scores per transcript - H1a)**

   - Both non-rubric (G1) and rubric-based (G2) methods demonstrated extremely high median consistency, with a median STDEV of 0.000 for the 50 scores assigned to each transcript. This indicates that, for at least half the transcripts, all 50 ratings under a given condition were identical.
   - The Mann-Whitney U test showed a statistically significant difference in the distributions of these STDEVs (U=882.0, p=0.00119).
   - While both medians were 0.0, G2 (Rubric-Based) had a larger Interquartile Range (IQR = 0.400) for its STDEVs compared to G1 (IQR = 0.000). This suggests that while both are highly consistent, G2 occasionally showed more variability in its 50 ratings for a given transcript than G1.
   - The hypothesis that G2 would have _lower_ STDEV (i.e., higher consistency) was **not supported**.

2. **Grading Accuracy (MAE vs. TargetTotalScore_Synthetic - H1b)**

   - The non-rubric method (G1) showed a lower median MAE (4.000) when comparing mean LLM scores to the synthetic target scores, compared to the rubric-based method (G2) which had a median MAE of 4.980.
   - The Wilcoxon signed-rank test indicated this difference in MAE is statistically significant (W=206.5, p=0.00207).
   - The hypothesis that G2 (Rubric-Based) would have _lower_ MAE (i.e., higher accuracy) was **not supported**. G1 (Non-Rubric) was found to be more accurate.

3. **Parsing and Data Integrity**

   - No parsing errors or missing scores were encountered across all 5000 grading attempts, indicating high technical reliability of the scoring process.

4. **Score Distribution (Preliminary - requires `exp1_summary_statistics.csv` review)**
   - Previous analysis (with fewer attempts and potentially different model) indicated both methods produced consistently high scores (16-20 range) with no scores below 16, suggesting potential ceiling effects. This needs to be re-verified with the current, larger dataset and specific model outputs.
   - The plot `exp1_mean_scores_vs_tts_distribution.png` should be reviewed to understand the overall score distributions for G1, G2, and TTS.

## Implications

### Strengths

1. **High Technical Reliability**: The system demonstrated perfect data capture with no parsing errors across 5000 assessments.
2. **High Intra-Transcript Consistency**: Both methods showed extremely high consistency in scores for individual transcripts (median STDEV of 0.0). This suggests the models, given a specific transcript and condition, produce very stable outputs over multiple runs.
3. **Measurable Accuracy**: The use of synthetic transcripts with target scores allowed for quantitative assessment of accuracy via MAE.
4. **Clearer Accuracy Signal**: The non-rubric method (G1) emerged as significantly more accurate (lower MAE) than the rubric-based method (G2) in matching synthetic target scores.

### Limitations

1. **Accuracy of Rubric-Based Method**: The rubric-based method (G2) was less accurate than the non-rubric method (G1). This might indicate issues with rubric interpretation by the LLM, the rubric's alignment with the synthetic data's implicit scoring, or the complexity of applying the rubric.
2. **Extreme Consistency - A Double-Edged Sword?**: While high consistency is good, a median STDEV of 0.0 might also suggest the model is "over-confident" or not sensitive to very subtle variations that might exist even within the same transcript if perceived slightly differently over runs. The G2 having a slightly wider IQR for STDEVs might indicate it captures a little more nuance or instability.
3. **Ceiling Effects (Preliminary)**: If prior findings of high scores and limited range hold true for this larger dataset, it could limit the system's ability to differentiate varying levels of communication quality, especially at the higher end. This needs verification from `exp1_summary_statistics.csv`.
4. **Generalizability to Real Data**: Findings are based on synthetic data. Performance on real-world transcripts (Experiment 2) will be crucial.

## Recommendations

1. **Investigate G2 (Rubric-Based) Accuracy**:
   - Analyze why G2 performed worse on accuracy (MAE). Review prompts for G2, and potentially the rubric itself for clarity and LLM amenability.
   - Examine transcripts where G2 had particularly high MAE.
2. **Verify Score Distributions**: Analyze `exp1_summary_statistics.csv` and relevant plots to confirm overall score distributions, means, medians, and check for ceiling effects or score compression with the current dataset. Update "Key Findings" and "Limitations" accordingly.
3. **Refine Non-Rubric Approach (G1)**: Given G1's better accuracy, explore if its interpretability can be enhanced, perhaps through chain-of-thought prompting for its rationale, without formally applying the full rubric.
4. **Future Experiments**: Proceed with real-data assessment (Experiment 2) to see if these patterns of consistency and accuracy hold. The comparison with human scores (Experiment 3) will be critical for validating either G1 or G2 (or a refined version).

## Conclusion

Experiment 1 successfully established a baseline for LLM-based assessment using synthetic data. Both non-rubric (G1) and rubric-based (G2) methods demonstrated high technical reliability and remarkable intra-transcript consistency. Counter to initial hypotheses, the non-rubric method (G1) proved to be significantly more accurate (lower MAE against synthetic target scores) than the rubric-based method (G2). While G2 showed slightly more variability in its per-transcript score STDEVs, both methods were extremely consistent.

These findings suggest that simpler, direct scoring (G1) might be more aligned with pre-defined quality targets in synthetic data than a more complex rubric-guided approach (G2), or that the current rubric/prompting for G2 needs refinement. The previously noted concerns about ceiling effects need re-evaluation with the full dataset. The results provide a solid, albeit different than expected, foundation for upcoming experiments with real-world data and human comparisons.

## Reflection and Analysis

### Overall Assessment

Experiment 1 provided crucial initial insights into our LLM-based health communication assessment system. Using synthetic transcripts with target scores allowed for a controlled evaluation of non-rubric (G1) vs. rubric-based (G2) grading. The results indicate high technical reliability and consistency for both methods, but challenge initial assumptions about the superiority of rubric-based grading in terms of accuracy against synthetic targets.

### Experiment Design Analysis

#### Strengths

1. **Controlled Environment & Scale**: Synthetic transcripts with embedded target scores provided a clear benchmark. The scale (50 transcripts, 5000 total assessments) allows for robust statistical analysis.
2. **Direct Comparison**: Testing G1 and G2 side-by-side with identical models (`gpt-4.1-mini`, temp 0.3) and data provides a clear comparison of the approaches.
3. **Quantitative Metrics**: Focus on MAE for accuracy and STDEV for consistency provided measurable outcomes.

#### Limitations and Considerations

1. **Synthetic Data Nature**: As always, synthetic data may not fully capture real-world complexities. The process used to generate transcripts and embed "target scores" might favor one assessment style.
2. **Interpretation of "Accuracy"**: MAE against synthetic targets is one form of accuracy. How this translates to meaningful assessment of real communication quality is still an open question for later experiments.
3. **Model Used for Generation vs. Assessment**: While different models (`GPT-4.1` for generation, `gpt-4.1-mini` for assessment) were used, they are still from the same family, which could lead to some shared biases or pattern recognition.

### Results Interpretation

#### Key Findings Analysis

1. **Consistency (STDEV per transcript)**: Both G1 and G2 are remarkably consistent (median STDEV = 0.0). G2 has a slightly higher IQR for these STDEVs, suggesting it might have more score spread for _some_ transcripts over its 50 runs, but the overall picture is high stability for both. This means the LLM gives nearly identical scores when rating the same transcript multiple times under the same condition.
2. **Accuracy (MAE vs. TTS)**: G1 (Non-Rubric) is significantly more accurate (Median MAE 4.000) than G2 (Rubric-Based, Median MAE 4.980). This is a key finding and suggests that the rubric, or its application by the LLM, might be misaligned with the inherent scoring of the synthetic data, or introduces noise.
3. **Previous Ceiling Effect/Score Compression**: The previous report noted ceiling effects (scores 16-20) and score compression with non-rubric. This needs re-evaluation using `exp1_summary_statistics.csv` from the current run to see if it persists. If so, it remains a concern for differentiating performance levels.
4. **Rubric-Based Performance**: The lower accuracy of G2 is notable. It implies that the structured rubric, in its current implementation, did not improve, and in fact, worsened alignment with the synthetic target scores compared to a more direct assessment.

### Critical Issues and Next Steps

#### Target Score Analysis - Progress and Further Needs

The calculation of MAE against `TargetTotalScore_Synthetic` is a crucial step forward. This directly addresses the previous gap.
However, further analysis is still needed:

- **Detailed Score Distribution Review**: A thorough review of `exp1_summary_statistics.csv` and the score distribution plots (`exp1_mean_scores_vs_tts_distribution.png`) is essential to understand overall score ranges, means, medians for G1 and G2, and to confirm if ceiling effects are present with this larger dataset and specific models.
- **Error Analysis for G2**: Deep dive into cases where G2 had high MAE. Is there a pattern? Are certain rubric items problematic?
- **Relationship between TTS and LLM Scores**: Beyond MAE, exploring the correlation and agreement between TTS and the scores from G1 and G2 would be beneficial.

### Recommendations for Future Work (updated)

1. **Immediate Priorities**
   - **Analyze `exp1_summary_statistics.csv` and Plots**: Update the report with findings on overall score distributions, means, medians, and ranges for G1 and G2. Confirm or refute previous ceiling effect observations.
   - **Investigate G2's Lower Accuracy**: Perform an error analysis on G2 assessments to understand why it was less accurate than G1.
2. **Methodological Improvements**
   - **Refine G2 Prompts/Rubric**: If G2 is to be pursued, its prompting or the rubric interaction needs refinement to improve accuracy.
   - **Explore G1 Enhancement**: Consider ways to make G1 more interpretable (e.g., requesting a rationale) without sacrificing its accuracy.
3. **Validation Steps**
   - Proceed with Experiment 2 (real data) and Experiment 3 (human comparison) to see how G1 and G2 perform in more realistic scenarios. The current findings make G1 a strong candidate if its performance holds.

### Future Directions

Remains largely the same: focus on real-world data, alternative prompting, different models, and human evaluation. The key shift is that G1 (non-rubric) now appears more promising based on synthetic data accuracy.

This reflection highlights that while the system is technically robust and consistent, the non-rubric approach showed superior accuracy on this synthetic dataset. The next critical step is to analyze the overall score distributions from the latest run and then validate these findings on real-world data.
