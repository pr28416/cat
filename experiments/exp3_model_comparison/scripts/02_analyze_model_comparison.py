#!/usr/bin/env python3
"""
Experiment 3: Model Comparison Analysis Script

This script analyzes the results from the model comparison experiment,
focusing on scoring consistency, model performance, and statistical comparisons.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
import scikit_posthocs as sp
from datetime import datetime

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- Constants ---
EXPERIMENT_ID = "EXP3_ModelComparison"
RESULTS_FILE = "exp3_modelcomparison_grading_results_curated_models.csv"
ALPHA = 0.05  # Significance level

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/processed_scores"
)
ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/analysis"
)


def load_and_validate_data():
    """Load the experiment results and perform basic validation."""
    print("=" * 60)
    print(f"🔍 LOADING AND VALIDATING DATA")
    print("=" * 60)

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    print(f"📊 Dataset Overview:")
    print(f"  Total records: {len(df)}")
    print(f"  Unique models: {df['model_name'].nunique()}")
    print(f"  Unique transcripts: {df['transcript_id'].nunique()}")
    print(f"  Date range: {df['experiment_id'].iloc[0]}")

    # Check for parsing errors
    successful_df = df[df["Parsing_Error"].isna()]
    error_rate = (len(df) - len(successful_df)) / len(df) * 100
    print(f"  Success rate: {len(successful_df)}/{len(df)} ({100-error_rate:.1f}%)")

    if error_rate > 0:
        print(f"  ⚠️  {error_rate:.1f}% parsing errors detected")

    # Model completion status
    print(f"\n📋 Completion by Model:")
    model_counts = df["model_name"].value_counts().sort_index()

    if not model_counts.empty:
        total_attempts_per_model = (
            df.groupby("model_name")["transcript_id"].nunique()
            * df.groupby(["model_name", "transcript_id"]).size().max()
        )

        for model, count in model_counts.items():
            expected_per_model = total_attempts_per_model.get(model, 0)
            completion_rate = (
                (count / expected_per_model * 100) if expected_per_model > 0 else 0
            )
            print(
                f"  {model}: {count}/{int(expected_per_model)} ({completion_rate:.1f}%)"
            )
    else:
        print("  No model data found.")

    return successful_df


def calculate_consistency_metrics(df):
    """Calculate consistency (STDEV) metrics for each model."""
    print("\n" + "=" * 60)
    print("🎯 CONSISTENCY ANALYSIS (Primary Metric)")
    print("=" * 60)

    consistency_data = []
    category_columns = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]

    # Calculate STDEV for each transcript-model combination for total and category scores
    for transcript in df["transcript_id"].unique():
        for model in df["model_name"].unique():
            subset = df[
                (df["transcript_id"] == transcript) & (df["model_name"] == model)
            ]
            if len(subset) >= 2:
                # Total Score
                total_scores = subset["Total_Score_Calculated"].values
                total_stdev = np.std(total_scores, ddof=1)
                mean_score = np.mean(total_scores)

                # Category Scores
                category_stdevs = {}
                for cat in category_columns:
                    cat_scores = subset[cat].values
                    category_stdevs[f"stdev_{cat}"] = np.std(cat_scores, ddof=1)

                record = {
                    "transcript_id": transcript,
                    "model_name": model,
                    "stdev": total_stdev,
                    "mean_score": mean_score,
                    "n_attempts": len(subset),
                    **category_stdevs,
                }
                consistency_data.append(record)

    consistency_df = pd.DataFrame(consistency_data)

    # Summary statistics by model
    print("📊 Consistency Summary (Standard Deviation of Total Scores):")
    print("   Lower values indicate more consistent scoring")
    print()

    model_stats = {}
    for model in sorted(consistency_df["model_name"].unique()):
        model_data = consistency_df[consistency_df["model_name"] == model]
        stdevs = model_data["stdev"]

        stats_dict = {
            "mean_stdev": stdevs.mean(),
            "median_stdev": stdevs.median(),
            "q75_stdev": stdevs.quantile(0.75),
            "q90_stdev": stdevs.quantile(0.90),
            "max_stdev": stdevs.max(),
            "n_transcripts": len(stdevs),
        }
        model_stats[model] = stats_dict

        print(f"  {model}:")
        print(f"    Mean STDEV: {stats_dict['mean_stdev']:.3f}")
        print(f"    Median STDEV: {stats_dict['median_stdev']:.3f}")
        print(f"    75th percentile: {stats_dict['q75_stdev']:.3f}")
        print(f"    90th percentile: {stats_dict['q90_stdev']:.3f}")
        print(f"    Max STDEV: {stats_dict['max_stdev']:.3f}")
        print(f"    N transcripts: {stats_dict['n_transcripts']}")
        print()

    # Ranking
    print("🏆 CONSISTENCY RANKING (Best to Worst):")
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["mean_stdev"])
    for i, (model, stats_dict) in enumerate(sorted_models, 1):
        print(f"  {i}. {model}: {stats_dict['mean_stdev']:.3f} mean STDEV")

    return consistency_df, model_stats


def perform_statistical_tests(consistency_df):
    """Perform statistical tests to compare model consistency."""
    print("\n" + "=" * 60)
    print("📈 STATISTICAL SIGNIFICANCE TESTS (CONSISTENCY)")
    print("=" * 60)

    models = sorted(consistency_df["model_name"].unique())
    transcripts = sorted(consistency_df["transcript_id"].unique())

    stdev_matrix = []
    valid_transcripts = []
    for transcript in transcripts:
        row_data = consistency_df[consistency_df["transcript_id"] == transcript]
        if len(row_data["model_name"].unique()) == len(models):
            stdevs = row_data.set_index("model_name").loc[models]["stdev"].values
            stdev_matrix.append(stdevs)
            valid_transcripts.append(transcript)

    stdev_matrix = np.array(stdev_matrix)

    print(f"📊 Test Setup:")
    print(f"  Models compared: {len(models)}")
    print(
        f"  Transcripts with complete data: {len(valid_transcripts)}/{len(transcripts)}"
    )
    print()

    pairwise_results_consistency = []
    if len(models) >= 3 and len(valid_transcripts) >= 5:
        print("🔬 Friedman Test (Non-parametric ANOVA for repeated measures on STDEV):")
        friedman_stat, friedman_p = friedmanchisquare(*stdev_matrix.T)
        print(f"   Friedman χ² = {friedman_stat:.4f}, p-value = {friedman_p:.6f}")

        if friedman_p < ALPHA:
            print(
                f"   ✅ SIGNIFICANT: Models differ significantly in consistency (p < {ALPHA})"
            )

            print("\n🔍 Post-hoc Pairwise Comparisons (Wilcoxon Signed-Rank on STDEV):")
            posthoc_df_values = sp.posthoc_wilcoxon(stdev_matrix, p_adjust="bonferroni")
            posthoc_df = pd.DataFrame(posthoc_df_values, index=models, columns=models)
            print("   P-values table (Bonferroni-corrected):")
            print(posthoc_df.round(6))

            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    p_val = posthoc_df.loc[model1, model2]

                    stdevs1 = stdev_matrix[:, i]
                    stdevs2 = stdev_matrix[:, j]

                    diff = stdevs1 - stdevs2
                    n_plus = np.sum(diff > 0)
                    n_minus = np.sum(diff < 0)
                    rc = (n_plus - n_minus) / len(diff) if len(diff) > 0 else 0

                    pairwise_results_consistency.append(
                        {
                            "model1": model1,
                            "model2": model2,
                            "p_value": p_val,
                            "significant": p_val
                            < (ALPHA / (len(models) * (len(models) - 1) / 2)),
                            "median_diff": np.median(stdevs1) - np.median(stdevs2),
                            "effect_size_rc": rc,
                        }
                    )
        else:
            print(
                f"   ❌ NOT SIGNIFICANT: No statistical difference in consistency found."
            )
    else:
        print("⚠️ Insufficient data for Friedman test on consistency.")

    return pairwise_results_consistency


def analyze_category_consistency(consistency_df):
    """Analyze consistency at the individual rubric category level."""
    print("\n" + "=" * 60)
    print("📊 CATEGORY-LEVEL CONSISTENCY ANALYSIS")
    print("=" * 60)

    models = sorted(consistency_df["model_name"].unique())
    category_columns = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]

    mean_stdevs_per_category = (
        consistency_df.groupby("model_name")[
            [f"stdev_{cat}" for cat in category_columns]
        ]
        .mean()
        .reset_index()
    )

    print("📊 Mean STDEV per category for each model:")
    print(mean_stdevs_per_category.round(3))

    # Visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    melted_df = pd.melt(
        mean_stdevs_per_category,
        id_vars=["model_name"],
        var_name="category",
        value_name="mean_stdev",
    )

    # Clean up category names for plotting
    melted_df["category"] = (
        melted_df["category"].str.replace("stdev_", "").str.replace("_", " ")
    )

    sns.barplot(x="category", y="mean_stdev", hue="model_name", data=melted_df, ax=ax)

    ax.set_title(
        f"{EXPERIMENT_ID}: Mean Standard Deviation by Rubric Category",
        fontsize=16,
        pad=20,
    )
    ax.set_xlabel("Rubric Category", fontsize=12)
    ax.set_ylabel("Mean Standard Deviation (Lower is Better)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot
    category_consistency_plot_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_category_consistency.png"
    )
    plt.savefig(category_consistency_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(
        f"\n✅ Saved category consistency plot to:\n   {category_consistency_plot_path}"
    )

    return mean_stdevs_per_category


def analyze_mean_score_differences(consistency_df):
    """Analyze for systemic differences in mean scores between models."""
    print("\n" + "=" * 60)
    print("⚖️ MEAN SCORE DIFFERENCE ANALYSIS")
    print("=" * 60)

    models = sorted(consistency_df["model_name"].unique())
    transcripts = sorted(consistency_df["transcript_id"].unique())

    mean_score_matrix = []
    valid_transcripts = []
    for transcript in transcripts:
        row_data = consistency_df[consistency_df["transcript_id"] == transcript]
        if len(row_data["model_name"].unique()) == len(models):
            scores = row_data.set_index("model_name").loc[models]["mean_score"].values
            mean_score_matrix.append(scores)
            valid_transcripts.append(transcript)

    mean_score_matrix = np.array(mean_score_matrix)

    pairwise_results_scores = []
    if len(models) >= 3 and len(valid_transcripts) >= 5:
        print("🔬 Friedman Test (for mean scores):")
        friedman_stat, friedman_p = friedmanchisquare(*mean_score_matrix.T)
        print(f"   Friedman χ² = {friedman_stat:.4f}, p-value = {friedman_p:.6f}")

        if friedman_p < ALPHA:
            print(
                f"   ✅ SIGNIFICANT: Models assign significantly different mean scores (p < {ALPHA})"
            )

            posthoc_df_values = sp.posthoc_wilcoxon(
                mean_score_matrix, p_adjust="bonferroni"
            )
            posthoc_df = pd.DataFrame(posthoc_df_values, index=models, columns=models)
            print("\n   P-values table for mean scores (Bonferroni-corrected):")
            print(posthoc_df.round(6))

            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    model1, model2 = models[i], models[j]
                    p_val = posthoc_df.loc[model1, model2]
                    pairwise_results_scores.append(
                        {
                            "model1": model1,
                            "model2": model2,
                            "p_value": p_val,
                            "significant": p_val
                            < (ALPHA / (len(models) * (len(models) - 1) / 2)),
                        }
                    )
        else:
            print(
                f"   ❌ NOT SIGNIFICANT: No significant systemic differences in mean scores found."
            )
    else:
        print("⚠️ Insufficient data for mean score difference test.")

    return pairwise_results_scores


def analyze_scoring_patterns(df):
    """Analyze overall scoring patterns and distributions."""
    print("\n" + "=" * 60)
    print("📊 SCORING PATTERNS ANALYSIS")
    print("=" * 60)

    print("🎯 Overall Score Statistics by Model:")
    for model in sorted(df["model_name"].unique()):
        model_data = df[df["model_name"] == model]
        scores = model_data["Total_Score_Calculated"]

        print(f"\n  {model}:")
        print(f"    Mean: {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"    Median: {scores.median():.2f}")
        print(f"    Range: {scores.min()}-{scores.max()}")
        print(f"    IQR: {scores.quantile(0.25):.1f}-{scores.quantile(0.75):.1f}")
        print(f"    N attempts: {len(scores)}")

    # Category-level analysis
    print(f"\n📋 Category-Level Score Analysis:")
    categories = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]

    for category in categories:
        print(f"\n  {category.replace('_', ' ')}:")
        for model in sorted(df["model_name"].unique()):
            model_data = df[df["model_name"] == model]
            cat_scores = model_data[category]
            print(f"    {model}: {cat_scores.mean():.2f} ± {cat_scores.std():.2f}")


def create_visualizations(df, consistency_df, model_stats):
    """Generate and save all plots for the analysis."""
    print("\n" + "=" * 60)
    print("📈 CREATING VISUALIZATIONS")
    print("=" * 60)

    # Ensure analysis directory exists
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Consistency comparison (boxplot)
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=consistency_df, x="model_name", y="stdev")
    plt.title(
        "Model Consistency Comparison\n(Standard Deviation of Total Scores per Transcript)",
        fontsize=14,
        fontweight="bold",
    )
    ax = plt.gca()
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Standard Deviation of Total Score (Lower is Better)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    consistency_plot_path = os.path.join(
        ANALYSIS_DIR, "exp3_consistency_comparison.png"
    )
    plt.savefig(consistency_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved consistency comparison plot to:\n   {consistency_plot_path}")

    # 2. Consistency Ranking Plot
    ranking_plot_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_consistency_ranking.png"
    )
    plt.figure(figsize=(12, 7))
    models_sorted = sorted(
        model_stats.items(), key=lambda x: x[1]["mean_stdev"]
    )  # best -> worst
    model_names = [item[0] for item in models_sorted]
    mean_stdevs = [item[1]["mean_stdev"] for item in models_sorted]

    # Friendly labels for paper-quality figures
    name_map = {
        "gpt-4.1-2025-04-14": "GPT-4.1",
        "gpt-4o-2024-08-06": "GPT-4o",
        "gpt-4o-mini-2024-07-18": "GPT-4o mini",
        "o3-2025-04-16": "o3",
        "o3-mini-2025-01-31": "o3 mini",
    }
    display_names = [name_map.get(n, n) for n in model_names]

    # Use model names on x-axis (categorical) and annotate values
    bars = plt.bar(display_names, mean_stdevs, alpha=0.9)
    plt.title(
        "Model Consistency: Mean Standard Deviation of Total Scores (lower is better)",
        fontsize=16,
        pad=20,
    )
    ax = plt.gca()
    ax.set_xlabel("Models (sorted by consistency)", fontsize=12)
    ax.set_ylabel("Mean STDEV of Total Score", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Color bars: green for best, red for worst
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.bar_label(
        bars,
        labels=[f"{v:.3f}" for v in mean_stdevs],
        padding=3,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(ranking_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved consistency ranking plot to:\n   {ranking_plot_path}")

    # 3. Score Distribution Plot
    score_dist_plot_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_score_distributions.png"
    )
    plt.figure(figsize=(12, 7))
    sns.kdeplot(
        data=df,
        x="Total_Score_Calculated",
        hue="model_name",
        fill=True,
        common_norm=False,
    )
    plt.title(
        f"{EXPERIMENT_ID}: Distribution of Total Scores by Model", fontsize=16, pad=20
    )
    ax = plt.gca()
    ax.set_xlabel("Total Score (Calculated)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(title="Model")
    plt.tight_layout()
    plt.savefig(score_dist_plot_path, dpi=300)
    plt.close()
    print(f"✅ Saved score distribution plot to:\n   {score_dist_plot_path}")


def save_summary_report(
    df,
    consistency_df,
    model_stats,
    pairwise_results_consistency,
    pairwise_results_scores,
    category_consistency_stats,
):
    """Save a comprehensive summary report of the analysis."""
    print("\n" + "=" * 60)
    print("💾 SAVING SUMMARY REPORT")
    print("=" * 60)

    summary_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_analysis_summary.txt"
    )

    report_content = []
    n_models = df["model_name"].nunique()
    n_transcripts = df["transcript_id"].nunique()

    report_content.append(f"EXPERIMENT 3: MODEL COMPARISON ANALYSIS REPORT\n")
    report_content.append("=" * 60 + "\n")
    report_content.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    report_content.append("DATASET OVERVIEW\n")
    report_content.append("-" * 20 + "\n")
    report_content.append(f"Total requests: {len(df)}\n")
    report_content.append(f"Models tested: {n_models}\n")
    report_content.append(f"Transcripts: {n_transcripts}\n")
    report_content.append(
        f"Success rate: {100 * (1 - df['Parsing_Error'].notna().mean()):.1f}%\n\n"
    )

    report_content.append("CONSISTENCY ANALYSIS RESULTS (Total Score)\n")
    report_content.append("-" * 50 + "\n")
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["mean_stdev"])
    report_content.append(
        "Ranking by Mean Standard Deviation (lower = more consistent):\n\n"
    )
    for i, (model, stats_dict) in enumerate(sorted_models, 1):
        report_content.append(f"{i}. {model}: {stats_dict['mean_stdev']:.3f}\n")
    report_content.append("\n")

    report_content.append("Detailed Statistics:\n\n")
    for model, stats_dict in sorted_models:
        report_content.append(f"{model}:\n")
        report_content.append(f"  Mean STDEV: {stats_dict['mean_stdev']:.3f}\n")
        report_content.append(f"  Median STDEV: {stats_dict['median_stdev']:.3f}\n")
        report_content.append(f"  90th percentile: {stats_dict['q90_stdev']:.3f}\n\n")

    report_content.append("STATISTICAL SIGNIFICANCE (CONSISTENCY)\n")
    report_content.append("-" * 50 + "\n")
    significant_pairs = [p for p in pairwise_results_consistency if p["significant"]]
    if significant_pairs:
        report_content.append(
            "Significant pairwise differences (Bonferroni corrected):\n\n"
        )
        for res in significant_pairs:
            report_content.append(
                f"  - {res['model1']} vs {res['model2']}: p = {res['p_value']:.6f}, effect size (rc) = {res['effect_size_rc']:.3f}\n"
            )
    else:
        report_content.append(
            "No significant pairwise differences found in consistency.\n"
        )
    report_content.append("\n")

    report_content.append("CATEGORY-LEVEL CONSISTENCY\n")
    report_content.append("-" * 50 + "\n")
    report_content.append("Mean Standard Deviation per Rubric Category:\n\n")
    report_content.append(category_consistency_stats.round(3).to_string() + "\n")
    report_content.append("\n\n")

    report_content.append("MEAN SCORE ANALYSIS\n")
    report_content.append("-" * 50 + "\n")
    report_content.append("Overall Score Patterns (Mean ± STDEV):\n")
    for model in sorted(df["model_name"].unique()):
        scores = df[df["model_name"] == model]["Total_Score_Calculated"]
        report_content.append(
            f"  {model}: {scores.mean():.2f} ± {scores.std():.2f} (range: {scores.min()}-{scores.max()})\n"
        )
    report_content.append("\n")

    report_content.append("Statistical Significance of Mean Score Differences:\n")
    significant_scores = [p for p in pairwise_results_scores if p["significant"]]
    if significant_scores:
        for res in significant_scores:
            report_content.append(
                f"  - {res['model1']} vs {res['model2']}: p = {res['p_value']:.6f}\n"
            )
    else:
        report_content.append(
            "  No significant differences found between models in mean scores.\n"
        )
    report_content.append("\n")

    try:
        with open(summary_path, "w") as f:
            f.write("\n".join(report_content))
        print(f"✅ Saved comprehensive summary report to:\n   {summary_path}")
    except IOError as e:
        print(f"❌ Error saving summary report: {e}")


def main():
    """Main function to run the analysis pipeline."""
    try:
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        df = load_and_validate_data()

        if df.empty:
            print("No data to analyze. Exiting.")
            return

        consistency_df, model_stats = calculate_consistency_metrics(df)
        pairwise_results_consistency = perform_statistical_tests(consistency_df)
        category_consistency_stats = analyze_category_consistency(consistency_df)
        pairwise_results_scores = analyze_mean_score_differences(consistency_df)

        analyze_scoring_patterns(df)
        create_visualizations(df, consistency_df, model_stats)

        save_summary_report(
            df,
            consistency_df,
            model_stats,
            pairwise_results_consistency,
            pairwise_results_scores,
            category_consistency_stats,
        )

        print("\n" + "=" * 60)
        print("🎉 ANALYSIS COMPLETE 🎉")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Please ensure the results file is in the correct directory.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
