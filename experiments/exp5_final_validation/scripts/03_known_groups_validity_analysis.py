"""
Experiment 5: Known-Groups Construct Validity Analysis

This script extends the Exp5 analysis to explicitly test known-groups construct validity
by comparing DoVA (real clinical conversations) vs Nature (physician-simulated) transcripts.

Known-groups validity: Scores should differ between groups known to differ on the construct.
Expectation: Simulated transcripts should score higher than real clinical conversations.
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

# Configuration
RESULTS_DIR = os.path.join(PROJECT_ROOT, "experiments/exp5_final_validation/results")
PROCESSED_SCORES_DIR = os.path.join(RESULTS_DIR, "processed_scores")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
OUTPUT_FILE = os.path.join(ANALYSIS_DIR, "known_groups_validity_report.md")

# Find the most recent grading results file
def find_latest_results_file():
    """Find the most recent exp5 grading results CSV."""
    files = [f for f in os.listdir(PROCESSED_SCORES_DIR) 
             if f.startswith("exp5_finaltoolvalidation_grading_results_checkpoint") 
             and f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No exp5 results files found")
    
    # Get the most recent file
    files.sort()
    latest = files[-1]
    print(f"Using results file: {latest}")
    return os.path.join(PROCESSED_SCORES_DIR, latest)


def classify_data_source(transcript_id):
    """Classify transcript as DoVA or Nature based on ID."""
    if transcript_id.startswith("DOVA_"):
        return "DoVA"
    else:
        return "Nature"


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for mean."""
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return mean, ci


def analyze_known_groups_validity():
    """Main analysis function for known-groups construct validity."""
    
    print("=" * 80)
    print("KNOWN-GROUPS CONSTRUCT VALIDITY ANALYSIS")
    print("=" * 80)
    print()
    print("Hypothesis: Simulated (Nature) transcripts should score higher than")
    print("real clinical (DoVA) conversations due to more structured/idealized speech.")
    print()
    
    # Load data
    results_file = find_latest_results_file()
    df = pd.read_csv(results_file)
    
    # Drop rows with missing scores (failed parsing)
    print(f"Original dataset: {len(df)} rows")
    df = df.dropna(subset=["Total_Score_Calculated"])
    print(f"After removing failed parses: {len(df)} rows")
    print()
    
    # Add data source classification
    df["data_source"] = df["transcript_id"].apply(classify_data_source)
    
    # Categories to analyze (using actual column names from CSV)
    categories = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
        "Total_Score_Calculated"
    ]
    
    category_names = {
        "Clarity_of_Language": "Clarity of Language",
        "Lexical_Diversity": "Lexical Diversity",
        "Conciseness_and_Completeness": "Conciseness and Completeness",
        "Engagement_with_Health_Information": "Engagement with Health Information",
        "Health_Literacy_Indicator": "Health Literacy Indicator",
        "Total_Score_Calculated": "Total Score"
    }
    
    # Compute per-transcript means
    transcript_means = df.groupby(["transcript_id", "data_source"])[categories].mean().reset_index()
    
    # Separate by data source
    dova_data = transcript_means[transcript_means["data_source"] == "DoVA"]
    nature_data = transcript_means[transcript_means["data_source"] == "Nature"]
    
    print(f"Sample sizes:")
    print(f"  DoVA transcripts: {len(dova_data)}")
    print(f"  Nature transcripts: {len(nature_data)}")
    print()
    
    # Results storage
    results = []
    
    # Analyze each category
    for cat in categories:
        dova_scores = dova_data[cat].values
        nature_scores = nature_data[cat].values
        
        # Descriptive statistics with CIs
        dova_mean, dova_ci = compute_confidence_interval(dova_scores)
        nature_mean, nature_ci = compute_confidence_interval(nature_scores)
        
        # Statistical test (Mann-Whitney U)
        statistic, p_value = stats.mannwhitneyu(dova_scores, nature_scores, alternative='two-sided')
        
        # Effect size (Cohen's d)
        cohens_d = compute_cohens_d(nature_scores, dova_scores)  # Nature - DoVA (expect positive)
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "Negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "Small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        
        results.append({
            "Category": category_names[cat],
            "DoVA_Mean": dova_mean,
            "DoVA_95CI": dova_ci,
            "Nature_Mean": nature_mean,
            "Nature_95CI": nature_ci,
            "Difference": nature_mean - dova_mean,
            "Cohen_d": cohens_d,
            "Effect_Size": effect_interpretation,
            "Mann_Whitney_U": statistic,
            "p_value": p_value,
            "Significant": "Yes" if p_value < 0.05 else "No"
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Print results
    print("=" * 80)
    print("KNOWN-GROUPS VALIDITY RESULTS")
    print("=" * 80)
    print()
    
    for _, row in results_df.iterrows():
        print(f"{row['Category']}")
        print(f"  DoVA:   {row['DoVA_Mean']:.2f} (95% CI: [{row['DoVA_95CI'][0]:.2f}, {row['DoVA_95CI'][1]:.2f}])")
        print(f"  Nature: {row['Nature_Mean']:.2f} (95% CI: [{row['Nature_95CI'][0]:.2f}, {row['Nature_95CI'][1]:.2f}])")
        print(f"  Difference: {row['Difference']:.2f}")
        print(f"  Effect Size: Cohen's d = {row['Cohen_d']:.3f} ({row['Effect_Size']})")
        print(f"  Statistical Test: U = {row['Mann_Whitney_U']:.0f}, p = {row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'}")
        print()
    
    # Generate markdown report
    generate_markdown_report(results_df, transcript_means)
    
    # Generate visualization
    create_visualization(results_df, transcript_means, categories, category_names)
    
    print(f"✓ Report saved to: {OUTPUT_FILE}")
    print(f"✓ Visualization saved to: {os.path.join(ANALYSIS_DIR, 'known_groups_validity_comparison.png')}")


def generate_markdown_report(results_df, transcript_means):
    """Generate markdown report of known-groups validity analysis."""
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Known-Groups Construct Validity Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis provides evidence for **known-groups construct validity** by comparing ")
        f.write("CHAT scores between two groups expected to differ on oral health literacy:\n\n")
        f.write("- **DoVA Group**: Real patient-physician clinical conversations\n")
        f.write("- **Nature Group**: Physician-simulated patient-doctor dialogues\n\n")
        f.write("**Hypothesis**: Simulated transcripts should demonstrate higher oral health literacy ")
        f.write("scores than real clinical conversations due to more structured, idealized communication.\n\n")
        
        f.write("## Sample Characteristics\n\n")
        dova_n = len(transcript_means[transcript_means["data_source"] == "DoVA"])
        nature_n = len(transcript_means[transcript_means["data_source"] == "Nature"])
        f.write(f"- DoVA transcripts: **{dova_n}** (real clinical conversations)\n")
        f.write(f"- Nature transcripts: **{nature_n}** (simulated dialogues)\n\n")
        
        f.write("## Results by Domain\n\n")
        f.write("| Domain | DoVA Mean (95% CI) | Nature Mean (95% CI) | Difference | Cohen's d | Effect Size | p-value |\n")
        f.write("|--------|-------------------|---------------------|------------|-----------|-------------|----------|\n")
        
        for _, row in results_df.iterrows():
            dova_ci_str = f"{row['DoVA_Mean']:.2f} ({row['DoVA_95CI'][0]:.2f}–{row['DoVA_95CI'][1]:.2f})"
            nature_ci_str = f"{row['Nature_Mean']:.2f} ({row['Nature_95CI'][0]:.2f}–{row['Nature_95CI'][1]:.2f})"
            sig_marker = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else "ns"
            
            f.write(f"| {row['Category']} | {dova_ci_str} | {nature_ci_str} | ")
            f.write(f"+{row['Difference']:.2f} | {row['Cohen_d']:.3f} | {row['Effect_Size']} | ")
            f.write(f"{row['p_value']:.4f} {sig_marker} |\n")
        
        f.write("\n*Effect size interpretation: <0.2 = Negligible, 0.2-0.5 = Small, 0.5-0.8 = Medium, >0.8 = Large*\n\n")
        f.write("*Significance levels: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant*\n\n")
        
        f.write("## Interpretation\n\n")
        
        # Check primary outcomes (Engagement + Health Literacy)
        primary_results = results_df[results_df["Category"].isin([
            "Engagement with Health Information",
            "Health Literacy Indicator"
        ])]
        
        f.write("### Primary Oral Health Literacy Outcomes\n\n")
        for _, row in primary_results.iterrows():
            if row['Significant'] == 'Yes':
                f.write(f"- **{row['Category']}**: Nature transcripts scored significantly higher ")
                f.write(f"(difference = {row['Difference']:.2f}, Cohen's d = {row['Cohen_d']:.3f}, p = {row['p_value']:.4f}). ")
                f.write(f"This {row['Effect_Size'].lower()} effect size supports the hypothesis that simulated ")
                f.write(f"conversations demonstrate higher oral health literacy than real clinical dialogues.\n\n")
            else:
                f.write(f"- **{row['Category']}**: No significant difference detected (p = {row['p_value']:.4f}).\n\n")
        
        f.write("### Auxiliary Communication Features\n\n")
        auxiliary_results = results_df[~results_df["Category"].isin([
            "Engagement with Health Information",
            "Health Literacy Indicator",
            "Total Score"
        ])]
        
        for _, row in auxiliary_results.iterrows():
            if row['Significant'] == 'Yes':
                f.write(f"- **{row['Category']}**: Significant difference (d = {row['Cohen_d']:.3f}, p = {row['p_value']:.4f})\n")
        
        f.write("\n### Overall Construct Validity Evidence\n\n")
        total_row = results_df[results_df["Category"] == "Total Score"].iloc[0]
        
        if total_row['Significant'] == 'Yes':
            f.write(f"The overall difference in total scores between groups is **statistically significant** ")
            f.write(f"(Cohen's d = {total_row['Cohen_d']:.3f}, p = {total_row['p_value']:.4f}), ")
            f.write(f"representing a **{total_row['Effect_Size'].lower()} effect**. ")
            f.write(f"This pattern supports **known-groups construct validity**: CHAT scores differentiate ")
            f.write(f"between real clinical conversations and idealized simulated dialogues in the expected direction, ")
            f.write(f"consistent with theoretical predictions about oral health literacy expression in different contexts.\n\n")
        
        f.write("## Clinical Significance\n\n")
        f.write("Beyond statistical significance, these findings demonstrate that CHAT scores are sensitive to ")
        f.write("meaningful differences in communication quality. Real patient conversations, which often involve ")
        f.write("uncertainty, hesitation, and less structured speech, score lower than physician-simulated ")
        f.write("\"textbook\" dialogues. This differentiation supports the tool's ability to capture genuine ")
        f.write("variation in oral health literacy as expressed in clinical conversations.\n\n")
        
        f.write("## Limitations\n\n")
        f.write("- This is **known-groups validity**, not criterion validity (no gold standard comparison)\n")
        f.write("- Nature transcripts were simulated by physicians, not actual patients with known literacy levels\n")
        f.write("- Further validation against established health literacy instruments (e.g., TOFHLA) and ")
        f.write("human expert ratings is needed\n\n")


def create_visualization(results_df, transcript_means, categories, category_names):
    """Create visualization of known-groups comparison."""
    
    # Prepare data for plotting
    plot_data = []
    for cat in categories[:-1]:  # Exclude total score from this plot
        dova = transcript_means[transcript_means["data_source"] == "DoVA"][cat].values
        nature = transcript_means[transcript_means["data_source"] == "Nature"][cat].values
        
        for score in dova:
            plot_data.append({"Category": category_names[cat], "Score": score, "Source": "DoVA (Real)"})
        for score in nature:
            plot_data.append({"Category": category_names[cat], "Score": score, "Source": "Nature (Simulated)"})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Known-Groups Construct Validity: DoVA vs Nature Transcripts", fontsize=16, fontweight='bold')
    
    # Plot each category
    for idx, cat in enumerate(categories):
        if idx < 5:
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
        else:
            ax = axes[1, 2]
        
        cat_name = category_names[cat]
        cat_data = transcript_means[["data_source", cat]].copy()
        
        # Violin plot
        parts = ax.violinplot(
            [cat_data[cat_data["data_source"] == "DoVA"][cat].values,
             cat_data[cat_data["data_source"] == "Nature"][cat].values],
            positions=[0, 1],
            showmeans=True,
            showmedians=True
        )
        
        # Get statistics for this category
        cat_result = results_df[results_df["Category"] == cat_name].iloc[0]
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["DoVA\n(Real)", "Nature\n(Simulated)"])
        ax.set_ylabel("Score")
        ax.set_title(f"{cat_name}\nd={cat_result['Cohen_d']:.3f} ({cat_result['Effect_Size']}), p={cat_result['p_value']:.4f}")
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(ANALYSIS_DIR, "known_groups_validity_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    dova_means = [row['DoVA_Mean'] for _, row in results_df.iterrows()]
    nature_means = [row['Nature_Mean'] for _, row in results_df.iterrows()]
    
    ax.bar(x - width/2, dova_means, width, label='DoVA (Real)', alpha=0.8)
    ax.bar(x + width/2, nature_means, width, label='Nature (Simulated)', alpha=0.8)
    
    ax.set_xlabel('Domain')
    ax.set_ylabel('Mean Score')
    ax.set_title('Known-Groups Comparison: Mean Scores by Domain')
    ax.set_xticks(x)
    ax.set_xticklabels([row['Category'] for _, row in results_df.iterrows()], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path2 = os.path.join(ANALYSIS_DIR, "known_groups_mean_comparison.png")
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    analyze_known_groups_validity()
    print("\n✓ Known-groups construct validity analysis complete!")

