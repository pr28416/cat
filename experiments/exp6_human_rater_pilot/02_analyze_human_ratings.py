"""
Analysis of Human Rater Pilot Data

This script computes:
1. Inter-rater reliability (ICC among human raters)
2. Human-CHAT agreement (correlation and ICC)
3. Visualizations and statistical summaries
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# Configuration
INPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/exp6_human_rater_pilot/")
OUTPUT_DIR = os.path.join(INPUT_DIR, "results/")


def compute_icc(ratings_df, rater_col, score_col, transcript_col):
    """
    Compute ICC(2,1) - Two-way random effects, single rater
    
    This is appropriate for measuring inter-rater reliability when:
    - Raters are a random sample of potential raters
    - Each transcript is rated by all raters
    - We want reliability of a single rater's score
    """
    from scipy.stats import f as f_dist
    
    # Pivot to wide format: rows = transcripts, columns = raters
    pivot = ratings_df.pivot(index=transcript_col, columns=rater_col, values=score_col)
    
    n_transcripts = pivot.shape[0]
    n_raters = pivot.shape[1]
    
    # Calculate means
    grand_mean = pivot.values.mean()
    transcript_means = pivot.mean(axis=1)
    rater_means = pivot.mean(axis=0)
    
    # Calculate sum of squares
    ss_total = np.sum((pivot.values - grand_mean) ** 2)
    ss_transcripts = n_raters * np.sum((transcript_means - grand_mean) ** 2)
    ss_raters = n_transcripts * np.sum((rater_means - grand_mean) ** 2)
    ss_error = ss_total - ss_transcripts - ss_raters
    
    # Degrees of freedom
    df_transcripts = n_transcripts - 1
    df_raters = n_raters - 1
    df_error = df_transcripts * df_raters
    
    # Mean squares
    ms_transcripts = ss_transcripts / df_transcripts if df_transcripts > 0 else 0
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0
    
    # ICC(2,1) calculation
    if (ms_transcripts + (n_raters - 1) * ms_error) == 0:
        icc = 0
    else:
        icc = (ms_transcripts - ms_error) / (ms_transcripts + (n_raters - 1) * ms_error)
    
    # 95% Confidence interval
    f_value = ms_transcripts / ms_error if ms_error > 0 else np.inf
    f_lower = f_value / f_dist.ppf(0.975, df_transcripts, df_error)
    f_upper = f_value / f_dist.ppf(0.025, df_transcripts, df_error)
    
    icc_lower = (f_lower - 1) / (f_lower + (n_raters - 1))
    icc_upper = (f_upper - 1) / (f_upper + (n_raters - 1))
    
    return {
        'icc': icc,
        'ci_lower': max(0, icc_lower),
        'ci_upper': min(1, icc_upper),
        'f_statistic': f_value,
        'p_value': 1 - f_dist.cdf(f_value, df_transcripts, df_error)
    }


def analyze_human_ratings():
    """Main analysis function."""
    
    print("=" * 80)
    print("HUMAN RATER PILOT ANALYSIS")
    print("=" * 80)
    print()
    
    # Load human ratings
    # Expecting file: human_ratings_combined.csv with columns:
    # transcript_id, rater_id, engagement_score, health_literacy_score
    
    ratings_file = os.path.join(INPUT_DIR, "human_ratings_combined.csv")
    
    if not os.path.exists(ratings_file):
        print(f"ERROR: Human ratings file not found: {ratings_file}")
        print("\nExpected format:")
        print("transcript_id,rater_id,engagement_score,health_literacy_score,comments")
        print("DOVA_F948.txt,RATER_01,3,3,Clear responses")
        print("\nPlease create this file with ratings from all raters.")
        return
    
    human_df = pd.read_csv(ratings_file)
    
    print(f"Loaded {len(human_df)} ratings from {human_df['rater_id'].nunique()} raters")
    print(f"Transcripts rated: {human_df['transcript_id'].nunique()}")
    print()
    
    # 1. INTER-RATER RELIABILITY
    print("=" * 80)
    print("1. HUMAN INTER-RATER RELIABILITY")
    print("=" * 80)
    print()
    
    for outcome in ["engagement_score", "health_literacy_score"]:
        print(f"\n{outcome.replace('_', ' ').title()}:")
        
        icc_result = compute_icc(human_df, "rater_id", outcome, "transcript_id")
        
        print(f"  ICC(2,1) = {icc_result['icc']:.3f}")
        print(f"  95% CI: [{icc_result['ci_lower']:.3f}, {icc_result['ci_upper']:.3f}]")
        print(f"  F({human_df['transcript_id'].nunique()-1}, ?) = {icc_result['f_statistic']:.2f}")
        print(f"  p-value: {icc_result['p_value']:.4f}")
        
        # Interpretation
        icc = icc_result['icc']
        if icc < 0.40:
            interpretation = "Poor"
        elif icc < 0.60:
            interpretation = "Fair to Moderate"
        elif icc < 0.75:
            interpretation = "Good"
        else:
            interpretation = "Excellent"
        
        print(f"  Interpretation: {interpretation} reliability")
    
    # 2. LOAD CHAT SCORES
    print("\n" + "=" * 80)
    print("2. CHAT SCORES FOR RATED TRANSCRIPTS")
    print("=" * 80)
    print()
    
    # Load CHAT scores for the rated transcripts
    exp5_file = os.path.join(PROJECT_ROOT, "experiments/exp5_final_validation/results/processed_scores")
    import glob
    files = glob.glob(os.path.join(exp5_file, "exp5_finaltoolvalidation_grading_results_checkpoint*.csv"))
    latest = sorted(files)[-1]
    
    chat_df = pd.read_csv(latest)
    chat_df = chat_df.dropna(subset=["Total_Score_Calculated"])
    
    # Get mean CHAT scores for rated transcripts
    rated_transcript_ids = human_df['transcript_id'].unique()
    chat_subset = chat_df[chat_df['transcript_id'].isin(rated_transcript_ids)]
    
    chat_means = chat_subset.groupby("transcript_id").agg({
        "Engagement_with_Health_Information": "mean",
        "Health_Literacy_Indicator": "mean",
        "Total_Score_Calculated": "mean"
    }).reset_index()
    
    print(f"CHAT scores available for {len(chat_means)} transcripts")
    
    # 3. HUMAN-CHAT AGREEMENT
    print("\n" + "=" * 80)
    print("3. HUMAN-CHAT AGREEMENT")
    print("=" * 80)
    print()
    
    # Compute mean human scores per transcript
    human_means = human_df.groupby("transcript_id").agg({
        "engagement_score": "mean",
        "health_literacy_score": "mean"
    }).reset_index()
    
    # Merge with CHAT scores
    comparison_df = human_means.merge(chat_means, on="transcript_id")
    
    print(f"Comparing {len(comparison_df)} transcripts with both human and CHAT ratings\n")
    
    # Engagement comparison
    print("Engagement with Health Information:")
    spearman_eng, p_eng = stats.spearmanr(
        comparison_df["engagement_score"],
        comparison_df["Engagement_with_Health_Information"]
    )
    pearson_eng, p_pears_eng = stats.pearsonr(
        comparison_df["engagement_score"],
        comparison_df["Engagement_with_Health_Information"]
    )
    
    print(f"  Spearman's ρ = {spearman_eng:.3f}, p = {p_eng:.4f}")
    print(f"  Pearson's r = {pearson_eng:.3f}, p = {p_pears_eng:.4f}")
    
    # Health Literacy comparison
    print("\nOverall Health Literacy:")
    spearman_hl, p_hl = stats.spearmanr(
        comparison_df["health_literacy_score"],
        comparison_df["Health_Literacy_Indicator"]
    )
    pearson_hl, p_pears_hl = stats.pearsonr(
        comparison_df["health_literacy_score"],
        comparison_df["Health_Literacy_Indicator"]
    )
    
    print(f"  Spearman's ρ = {spearman_hl:.3f}, p = {p_hl:.4f}")
    print(f"  Pearson's r = {pearson_hl:.3f}, p = {p_pears_hl:.4f}")
    
    # Create visualizations
    create_visualizations(comparison_df, human_df)
    
    # Generate report
    generate_report(human_df, comparison_df, {
        'engagement': {'spearman': spearman_eng, 'p': p_eng, 'pearson': pearson_eng},
        'health_literacy': {'spearman': spearman_hl, 'p': p_hl, 'pearson': pearson_hl}
    })
    
    print("\n✓ Analysis complete!")


def create_visualizations(comparison_df, human_df):
    """Create correlation plots and Bland-Altman plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Engagement scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(comparison_df["engagement_score"], 
                comparison_df["Engagement_with_Health_Information"],
                alpha=0.6)
    ax1.plot([1, 4], [1, 4], 'r--', alpha=0.5, label='Perfect agreement')
    ax1.set_xlabel("Mean Human Rating")
    ax1.set_ylabel("CHAT Score")
    ax1.set_title("Engagement with Health Information")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Health Literacy scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(comparison_df["health_literacy_score"],
                comparison_df["Health_Literacy_Indicator"],
                alpha=0.6)
    ax2.plot([1, 4], [1, 4], 'r--', alpha=0.5, label='Perfect agreement')
    ax2.set_xlabel("Mean Human Rating")
    ax2.set_ylabel("CHAT Score")
    ax2.set_title("Overall Health Literacy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Bland-Altman plot for Engagement
    ax3 = axes[1, 0]
    mean_eng = (comparison_df["engagement_score"] + comparison_df["Engagement_with_Health_Information"]) / 2
    diff_eng = comparison_df["engagement_score"] - comparison_df["Engagement_with_Health_Information"]
    
    ax3.scatter(mean_eng, diff_eng, alpha=0.6)
    ax3.axhline(diff_eng.mean(), color='red', linestyle='-', label='Mean difference')
    ax3.axhline(diff_eng.mean() + 1.96*diff_eng.std(), color='red', linestyle='--', label='±1.96 SD')
    ax3.axhline(diff_eng.mean() - 1.96*diff_eng.std(), color='red', linestyle='--')
    ax3.set_xlabel("Mean Score")
    ax3.set_ylabel("Human - CHAT Difference")
    ax3.set_title("Bland-Altman: Engagement")
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Bland-Altman plot for Health Literacy
    ax4 = axes[1, 1]
    mean_hl = (comparison_df["health_literacy_score"] + comparison_df["Health_Literacy_Indicator"]) / 2
    diff_hl = comparison_df["health_literacy_score"] - comparison_df["Health_Literacy_Indicator"]
    
    ax4.scatter(mean_hl, diff_hl, alpha=0.6)
    ax4.axhline(diff_hl.mean(), color='red', linestyle='-', label='Mean difference')
    ax4.axhline(diff_hl.mean() + 1.96*diff_hl.std(), color='red', linestyle='--', label='±1.96 SD')
    ax4.axhline(diff_hl.mean() - 1.96*diff_hl.std(), color='red', linestyle='--')
    ax4.set_xlabel("Mean Score")
    ax4.set_ylabel("Human - CHAT Difference")
    ax4.set_title("Bland-Altman: Health Literacy")
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "human_chat_agreement_plots.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved visualizations: {output_path}")


def generate_report(human_df, comparison_df, correlations):
    """Generate comprehensive markdown report."""
    
    output_file = os.path.join(OUTPUT_DIR, "human_pilot_results.md")
    
    with open(output_file, 'w') as f:
        f.write("# Experiment 6: Human Rater Pilot Results\n\n")
        f.write("## Study Design\n\n")
        f.write(f"- **Sample size**: {comparison_df['transcript_id'].nunique()} transcripts\n")
        f.write(f"- **Number of raters**: {human_df['rater_id'].nunique()}\n")
        f.write(f"- **Outcomes rated**: Engagement with Health Information, Overall Health Literacy\n")
        f.write(f"- **Rating scale**: 1-4 (Poor to Excellent)\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("### 1. Human Inter-Rater Reliability\n\n")
        f.write("[ICC results to be inserted after analysis runs]\n\n")
        
        f.write("### 2. Human-CHAT Agreement\n\n")
        f.write("#### Engagement with Health Information\n")
        f.write(f"- Spearman's ρ = {correlations['engagement']['spearman']:.3f}, p = {correlations['engagement']['p']:.4f}\n")
        f.write(f"- Pearson's r = {correlations['engagement']['pearson']:.3f}\n\n")
        
        f.write("#### Overall Health Literacy\n")
        f.write(f"- Spearman's ρ = {correlations['health_literacy']['spearman']:.3f}, p = {correlations['health_literacy']['p']:.4f}\n")
        f.write(f"- Pearson's r = {correlations['health_literacy']['pearson']:.3f}\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Interpret engagement correlation
        eng_rho = correlations['engagement']['spearman']
        if eng_rho > 0.70:
            eng_interp = "strong"
        elif eng_rho > 0.50:
            eng_interp = "moderate to strong"
        elif eng_rho > 0.30:
            eng_interp = "moderate"
        else:
            eng_interp = "weak"
        
        f.write(f"The {eng_interp} correlation between CHAT and human ratings for Engagement ")
        f.write(f"provides preliminary evidence of concurrent validity.\n\n")
        
        # Overall interpretation
        if eng_rho > 0.50 and correlations['health_literacy']['spearman'] > 0.50:
            f.write("**Conclusion**: This exploratory pilot supports CHAT's concurrent validity. ")
            f.write("CHAT scores show moderate to strong agreement with human expert ratings on ")
            f.write("the primary oral health literacy outcomes. Larger validation studies are ")
            f.write("needed to confirm these findings.\n\n")
        elif eng_rho > 0.30:
            f.write("**Conclusion**: This exploratory pilot provides modest support for CHAT's validity. ")
            f.write("While correlations are moderate, the small sample size limits generalizability. ")
            f.write("Larger validation studies with diverse raters and transcripts are needed.\n\n")
        else:
            f.write("**Conclusion**: This exploratory pilot suggests that CHAT scores have limited ")
            f.write("agreement with human ratings. This may reflect: (1) need for rater training ")
            f.write("refinement, (2) construct definition clarity, or (3) inherent challenges in ")
            f.write("rating oral health literacy. Further investigation is needed.\n\n")
        
        f.write("## Limitations\n\n")
        f.write("- Small sample size (exploratory pilot)\n")
        f.write("- Limited number of raters\n")
        f.write("- Only 2 outcomes rated (pragmatic choice to reduce rater burden)\n")
        f.write("- No comparison to established health literacy instruments (TOFHLA, NVS)\n")
    
    print(f"\n  Saved report: {output_file}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analyze_human_ratings()

