#!/usr/bin/env python3
"""
Experiment 5: Final Tool Validation Analysis

This script analyzes the results from the final validation experiment,
focusing on:
1. Scoring consistency (STDEV analysis)
2. Benchmark performance against PRD success metrics
3. Uncertainty quantification exploration
4. Descriptive statistics and visualizations

Success Metrics from PRD:
- Mean STDEV of total scores < 1.0
- 95% of transcripts achieving acceptable consistency
- Characterization of confidence/uncertainty patterns
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- Constants ---
EXPERIMENT_ID = "EXP5_FinalToolValidation"
TARGET_STDEV_THRESHOLD = 1.0  # Success metric from PRD
CATEGORY_STDEV_THRESHOLD = 0.5  # Category-level threshold
SUCCESS_PERCENTAGE_TARGET = 95  # 95% of transcripts should meet threshold

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp5_final_validation/results/processed_scores"
)
ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp5_final_validation/results/analysis"
)
RESULTS_FILE = f"{EXPERIMENT_ID.lower()}_grading_results.csv"

# Ensure analysis directory exists
os.makedirs(ANALYSIS_DIR, exist_ok=True)


def load_and_validate_data():
    """Load and validate the experiment results."""
    print("=" * 80)
    print("📊 LOADING AND VALIDATING EXPERIMENT 5 DATA")
    print("=" * 80)

    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")

    df = pd.read_csv(results_path)

    print(f"📋 Dataset Overview:")
    print(f"   Total records: {len(df):,}")
    print(f"   Unique transcripts: {df['transcript_id'].nunique():,}")
    print(f"   Model used: {df['model_name'].iloc[0]}")
    print(f"   Prompt strategy: {df['prompt_strategy'].iloc[0]}")
    print(f"   Temperature: {df['temperature'].iloc[0]}")

    # Check data quality
    successful_df = df[df["Parsing_Error"].isna()]
    error_rate = (len(df) - len(successful_df)) / len(df) * 100
    print(
        f"   Success rate: {len(successful_df):,}/{len(df):,} ({100-error_rate:.1f}%)"
    )

    if error_rate > 5:
        print(f"   ⚠️  High error rate detected: {error_rate:.1f}%")

    # Check attempts per transcript
    attempts_per_transcript = df.groupby("transcript_id").size()
    expected_attempts = 20  # Updated configuration: 20 attempts per transcript
    complete_transcripts = (attempts_per_transcript == expected_attempts).sum()

    print(f"   Expected attempts per transcript: {expected_attempts}")
    print(
        f"   Transcripts with complete data: {complete_transcripts:,}/{df['transcript_id'].nunique():,}"
    )
    print(f"   Mean attempts per transcript: {attempts_per_transcript.mean():.1f}")

    return successful_df


def calculate_consistency_metrics(df):
    """Calculate STDEV metrics for total scores and individual categories."""
    print("\n" + "=" * 80)
    print("📏 CALCULATING CONSISTENCY METRICS")
    print("=" * 80)

    # Categories to analyze
    categories = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]

    consistency_results = []

    # Calculate STDEV for each transcript
    for transcript_id in df["transcript_id"].unique():
        transcript_data = df[df["transcript_id"] == transcript_id]

        if len(transcript_data) < 2:  # Need at least 2 attempts for STDEV
            continue

        result = {
            "transcript_id": transcript_id,
            "num_attempts": len(transcript_data),
            "stdev_total_score": transcript_data["Total_Score_Calculated"].std(),
            "mean_total_score": transcript_data["Total_Score_Calculated"].mean(),
        }

        # Category-level STDEV
        for category in categories:
            if category in transcript_data.columns:
                result[f"stdev_{category}"] = transcript_data[category].std()
                result[f"mean_{category}"] = transcript_data[category].mean()

        consistency_results.append(result)

    consistency_df = pd.DataFrame(consistency_results)

    print(f"📊 Consistency Analysis Summary:")
    print(f"   Transcripts analyzed: {len(consistency_df):,}")
    print(
        f"   Mean STDEV (total score): {consistency_df['stdev_total_score'].mean():.3f}"
    )
    print(
        f"   Median STDEV (total score): {consistency_df['stdev_total_score'].median():.3f}"
    )
    print(
        f"   90th percentile STDEV: {consistency_df['stdev_total_score'].quantile(0.9):.3f}"
    )
    print(f"   Max STDEV: {consistency_df['stdev_total_score'].max():.3f}")

    return consistency_df


def evaluate_benchmark_performance(consistency_df):
    """Evaluate performance against PRD success metrics."""
    print("\n" + "=" * 80)
    print("🎯 BENCHMARK PERFORMANCE EVALUATION")
    print("=" * 80)

    # Primary success metric: Mean STDEV < 1.0
    mean_stdev = consistency_df["stdev_total_score"].mean()
    benchmark_met = mean_stdev < TARGET_STDEV_THRESHOLD

    print(f"📋 Primary Success Metric (PRD Section 7.5.3):")
    print(f"   Target: Mean STDEV < {TARGET_STDEV_THRESHOLD}")
    print(f"   Achieved: {mean_stdev:.3f}")
    print(f"   Status: {'✅ PASSED' if benchmark_met else '❌ FAILED'}")

    # Percentage of transcripts meeting threshold
    transcripts_meeting_threshold = (
        consistency_df["stdev_total_score"] < TARGET_STDEV_THRESHOLD
    ).sum()
    percentage_meeting = transcripts_meeting_threshold / len(consistency_df) * 100
    percentage_benchmark_met = percentage_meeting >= SUCCESS_PERCENTAGE_TARGET

    print(f"\n📊 Coverage Success Metric:")
    print(
        f"   Target: {SUCCESS_PERCENTAGE_TARGET}% of transcripts with STDEV < {TARGET_STDEV_THRESHOLD}"
    )
    print(
        f"   Achieved: {transcripts_meeting_threshold:,}/{len(consistency_df):,} ({percentage_meeting:.1f}%)"
    )
    print(f"   Status: {'✅ PASSED' if percentage_benchmark_met else '❌ FAILED'}")

    # Category-level analysis
    categories = [
        col
        for col in consistency_df.columns
        if col.startswith("stdev_") and col != "stdev_total_score"
    ]

    print(f"\n📋 Category-Level Performance:")
    category_results = {}
    for category in categories:
        cat_name = category.replace("stdev_", "").replace("_", " ")
        mean_cat_stdev = consistency_df[category].mean()
        cat_threshold_met = (consistency_df[category] < CATEGORY_STDEV_THRESHOLD).sum()
        cat_percentage = cat_threshold_met / len(consistency_df) * 100

        category_results[cat_name] = {
            "mean_stdev": mean_cat_stdev,
            "percentage_under_threshold": cat_percentage,
        }

        print(f"   {cat_name}:")
        print(f"     Mean STDEV: {mean_cat_stdev:.3f}")
        print(f"     Under {CATEGORY_STDEV_THRESHOLD}: {cat_percentage:.1f}%")

    # Overall assessment
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if benchmark_met and percentage_benchmark_met:
        print(f"   ✅ EXPERIMENT 5 SUCCESS: All primary benchmarks achieved")
        print(f"   🎉 The optimized tool meets PRD success criteria")
    elif benchmark_met:
        print(
            f"   ⚠️  PARTIAL SUCCESS: Mean STDEV target met, but coverage below target"
        )
    elif percentage_benchmark_met:
        print(
            f"   ⚠️  PARTIAL SUCCESS: Coverage target met, but mean STDEV above target"
        )
    else:
        print(f"   ❌ EXPERIMENT 5 FAILURE: Primary benchmarks not achieved")

    return {
        "mean_stdev": mean_stdev,
        "benchmark_met": benchmark_met,
        "percentage_meeting": percentage_meeting,
        "percentage_benchmark_met": percentage_benchmark_met,
        "category_results": category_results,
    }


def analyze_score_distributions(df, consistency_df):
    """Analyze the distribution of scores assigned by the optimized tool."""
    print("\n" + "=" * 80)
    print("📊 SCORE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Total score statistics
    total_scores = df["Total_Score_Calculated"]
    print(f"📈 Total Score Statistics:")
    print(f"   Mean: {total_scores.mean():.2f} ± {total_scores.std():.2f}")
    print(f"   Median: {total_scores.median():.2f}")
    print(f"   Range: {total_scores.min()}-{total_scores.max()}")
    print(
        f"   IQR: {total_scores.quantile(0.25):.1f}-{total_scores.quantile(0.75):.1f}"
    )

    # Category-level statistics
    categories = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]

    print(f"\n📋 Category-Level Score Statistics:")
    for category in categories:
        if category in df.columns:
            scores = df[category]
            print(f"   {category.replace('_', ' ')}:")
            print(f"     Mean: {scores.mean():.2f} ± {scores.std():.2f}")
            print(f"     Range: {scores.min()}-{scores.max()}")

    return {
        "total_score_stats": {
            "mean": total_scores.mean(),
            "std": total_scores.std(),
            "median": total_scores.median(),
            "min": total_scores.min(),
            "max": total_scores.max(),
        }
    }


def analyze_data_source_differences(df, consistency_df):
    """Compare performance between DoVA and Nature paper transcripts."""
    print("\n" + "=" * 80)
    print("🔍 DATA SOURCE COMPARISON (DoVA vs Nature)")
    print("=" * 80)

    # Add data source classification
    consistency_df["data_source"] = consistency_df["transcript_id"].apply(
        lambda x: "DoVA" if x.startswith("DOVA_") else "Nature"
    )

    # Count transcripts by source
    source_counts = consistency_df["data_source"].value_counts()
    print(f"📊 Transcript Distribution by Source:")
    for source, count in source_counts.items():
        print(
            f"   {source}: {count} transcripts ({count/len(consistency_df)*100:.1f}%)"
        )

    # Compare consistency (STDEV) between sources
    print(f"\n📏 Consistency Comparison (STDEV of Total Scores):")

    dova_stdev = consistency_df[consistency_df["data_source"] == "DoVA"][
        "stdev_total_score"
    ]
    nature_stdev = consistency_df[consistency_df["data_source"] == "Nature"][
        "stdev_total_score"
    ]

    print(f"   DoVA Transcripts:")
    print(f"     Mean STDEV: {dova_stdev.mean():.3f} ± {dova_stdev.std():.3f}")
    print(f"     Median STDEV: {dova_stdev.median():.3f}")
    print(f"     Range: {dova_stdev.min():.3f} - {dova_stdev.max():.3f}")

    print(f"   Nature Transcripts:")
    print(f"     Mean STDEV: {nature_stdev.mean():.3f} ± {nature_stdev.std():.3f}")
    print(f"     Median STDEV: {nature_stdev.median():.3f}")
    print(f"     Range: {nature_stdev.min():.3f} - {nature_stdev.max():.3f}")

    # Statistical test for difference
    from scipy.stats import mannwhitneyu

    statistic, p_value = mannwhitneyu(dova_stdev, nature_stdev, alternative="two-sided")

    print(f"\n📈 Statistical Test (Mann-Whitney U):")
    print(f"   Test statistic: {statistic:.2f}")
    print(f"   P-value: {p_value:.4f}")
    print(
        f"   Significance: {'✅ Significant' if p_value < 0.05 else '❌ Not significant'} (α = 0.05)"
    )

    if p_value < 0.05:
        better_source = "DoVA" if dova_stdev.mean() < nature_stdev.mean() else "Nature"
        print(
            f"   Result: {better_source} transcripts show significantly better consistency"
        )
    else:
        print(f"   Result: No significant difference in consistency between sources")

    # Compare mean scores between sources
    print(f"\n📊 Mean Score Comparison:")

    dova_mean_scores = consistency_df[consistency_df["data_source"] == "DoVA"][
        "mean_total_score"
    ]
    nature_mean_scores = consistency_df[consistency_df["data_source"] == "Nature"][
        "mean_total_score"
    ]

    print(
        f"   DoVA Mean Scores: {dova_mean_scores.mean():.2f} ± {dova_mean_scores.std():.2f}"
    )
    print(
        f"   Nature Mean Scores: {nature_mean_scores.mean():.2f} ± {nature_mean_scores.std():.2f}"
    )

    # Statistical test for score differences
    score_statistic, score_p_value = mannwhitneyu(
        dova_mean_scores, nature_mean_scores, alternative="two-sided"
    )
    print(f"   Score difference p-value: {score_p_value:.4f}")
    print(
        f"   Score difference: {'✅ Significant' if score_p_value < 0.05 else '❌ Not significant'}"
    )

    # Benchmark achievement by source
    print(f"\n🎯 Benchmark Achievement by Source:")

    TARGET_STDEV_THRESHOLD = 1.0

    dova_under_threshold = (dova_stdev < TARGET_STDEV_THRESHOLD).sum()
    nature_under_threshold = (nature_stdev < TARGET_STDEV_THRESHOLD).sum()

    dova_percentage = dova_under_threshold / len(dova_stdev) * 100
    nature_percentage = nature_under_threshold / len(nature_stdev) * 100

    print(
        f"   DoVA: {dova_under_threshold}/{len(dova_stdev)} ({dova_percentage:.1f}%) under threshold"
    )
    print(
        f"   Nature: {nature_under_threshold}/{len(nature_stdev)} ({nature_percentage:.1f}%) under threshold"
    )

    return {
        "source_comparison": {
            "dova_count": len(dova_stdev),
            "nature_count": len(nature_stdev),
            "dova_mean_stdev": dova_stdev.mean(),
            "nature_mean_stdev": nature_stdev.mean(),
            "stdev_p_value": p_value,
            "dova_mean_score": dova_mean_scores.mean(),
            "nature_mean_score": nature_mean_scores.mean(),
            "score_p_value": score_p_value,
            "dova_benchmark_percentage": dova_percentage,
            "nature_benchmark_percentage": nature_percentage,
        }
    }


def explore_uncertainty_quantification(df, consistency_df):
    """Explore methods for quantifying score uncertainty/confidence."""
    print("\n" + "=" * 80)
    print("🔍 UNCERTAINTY QUANTIFICATION EXPLORATION")
    print("=" * 80)

    print(f"📊 Variance-Based Uncertainty Analysis:")

    # Relationship between STDEV and other factors
    print(f"\n🔗 Factors Potentially Related to Uncertainty:")

    # 1. Score level vs uncertainty
    score_uncertainty_corr = stats.spearmanr(
        consistency_df["mean_total_score"], consistency_df["stdev_total_score"]
    )
    print(
        f"   Score level ↔ Uncertainty: ρ = {score_uncertainty_corr.correlation:.3f}, p = {score_uncertainty_corr.pvalue:.3f}"
    )

    # 2. Transcript length vs uncertainty (if available)
    # This would require transcript metadata

    # 3. High vs low uncertainty transcripts
    high_uncertainty = consistency_df["stdev_total_score"].quantile(0.9)
    low_uncertainty = consistency_df["stdev_total_score"].quantile(0.1)

    print(f"\n📊 Uncertainty Distribution:")
    print(f"   10th percentile (low uncertainty): {low_uncertainty:.3f}")
    print(f"   90th percentile (high uncertainty): {high_uncertainty:.3f}")
    print(
        f"   Uncertainty ratio (90th/10th): {high_uncertainty/low_uncertainty if low_uncertainty > 0 else 'inf':.1f}"
    )

    # Identify high uncertainty cases
    high_uncertainty_transcripts = consistency_df[
        consistency_df["stdev_total_score"] > high_uncertainty
    ]["transcript_id"].tolist()

    print(f"   High uncertainty transcripts: {len(high_uncertainty_transcripts)}")
    if len(high_uncertainty_transcripts) <= 5:
        print(f"   Examples: {high_uncertainty_transcripts}")

    return {
        "score_uncertainty_correlation": score_uncertainty_corr.correlation,
        "high_uncertainty_threshold": high_uncertainty,
        "high_uncertainty_transcripts": high_uncertainty_transcripts,
    }


def create_visualizations(df, consistency_df, benchmark_results):
    """Generate comprehensive visualizations for the analysis."""
    print("\n" + "=" * 80)
    print("📈 CREATING VISUALIZATIONS")
    print("=" * 80)

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. STDEV Distribution (Primary Metric)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(consistency_df["stdev_total_score"], bins=30, kde=True)
    plt.axvline(
        TARGET_STDEV_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Target Threshold ({TARGET_STDEV_THRESHOLD})",
    )
    plt.axvline(
        consistency_df["stdev_total_score"].mean(),
        color="green",
        linestyle="-",
        label=f'Mean ({consistency_df["stdev_total_score"].mean():.3f})',
    )
    plt.title("Distribution of Score Standard Deviations\n(Primary Success Metric)")
    plt.xlabel("Standard Deviation of Total Score")
    plt.ylabel("Count")
    plt.legend()

    # 2. Score Distribution
    plt.subplot(2, 2, 2)
    sns.histplot(df["Total_Score_Calculated"], bins=20, kde=True)
    plt.title("Distribution of Total Scores\n(All Assessments)")
    plt.xlabel("Total Score (5-20)")
    plt.ylabel("Count")

    # 3. Consistency vs Mean Score
    plt.subplot(2, 2, 3)
    plt.scatter(
        consistency_df["mean_total_score"],
        consistency_df["stdev_total_score"],
        alpha=0.6,
    )
    plt.axhline(TARGET_STDEV_THRESHOLD, color="red", linestyle="--", alpha=0.7)
    plt.title("Score Consistency vs Mean Score")
    plt.xlabel("Mean Total Score")
    plt.ylabel("Standard Deviation")

    # 4. Category-level consistency
    plt.subplot(2, 2, 4)
    category_cols = [
        col
        for col in consistency_df.columns
        if col.startswith("stdev_") and col != "stdev_total_score"
    ]
    category_means = [consistency_df[col].mean() for col in category_cols]
    category_names = [
        col.replace("stdev_", "").replace("_", "\n") for col in category_cols
    ]

    bars = plt.bar(range(len(category_names)), category_means)
    plt.axhline(
        CATEGORY_STDEV_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Category Threshold ({CATEGORY_STDEV_THRESHOLD})",
    )
    plt.title("Mean STDEV by Category")
    plt.xlabel("Rubric Category")
    plt.ylabel("Mean Standard Deviation")
    plt.xticks(range(len(category_names)), category_names, rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()

    # Save the comprehensive plot
    main_plot_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_comprehensive_analysis.png"
    )
    plt.savefig(main_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved comprehensive analysis plot: {main_plot_path}")

    # 2. Detailed STDEV distribution with benchmark overlay
    plt.figure(figsize=(10, 6))
    sns.histplot(consistency_df["stdev_total_score"], bins=50, kde=True, alpha=0.7)
    plt.axvline(
        TARGET_STDEV_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Success Threshold ({TARGET_STDEV_THRESHOLD})",
    )
    plt.axvline(
        consistency_df["stdev_total_score"].mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f'Achieved Mean ({consistency_df["stdev_total_score"].mean():.3f})',
    )

    # Add benchmark status text
    status_text = (
        "✅ BENCHMARK PASSED"
        if benchmark_results["benchmark_met"]
        else "❌ BENCHMARK FAILED"
    )
    plt.text(
        0.7,
        0.9,
        status_text,
        transform=plt.gca().transAxes,
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor=(
                "lightgreen" if benchmark_results["benchmark_met"] else "lightcoral"
            ),
        ),
    )

    plt.title(
        f"{EXPERIMENT_ID}: Final Tool Consistency Performance\n"
        f'({benchmark_results["percentage_meeting"]:.1f}% of transcripts under threshold)',
        fontsize=14,
    )
    plt.xlabel("Standard Deviation of Total Score (Lower = More Consistent)")
    plt.ylabel("Number of Transcripts")
    plt.legend()

    stdev_plot_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_stdev_distribution.png"
    )
    plt.savefig(stdev_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved STDEV distribution plot: {stdev_plot_path}")


def save_comprehensive_report(
    df,
    consistency_df,
    benchmark_results,
    score_stats,
    source_comparison,
    uncertainty_results,
):
    """Save a comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("💾 SAVING COMPREHENSIVE REPORT")
    print("=" * 80)

    report_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_analysis_report.md"
    )

    with open(report_path, "w") as f:
        f.write(f"# Experiment 5: Final Tool Validation Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Executive Summary\n\n")

        if (
            benchmark_results["benchmark_met"]
            and benchmark_results["percentage_benchmark_met"]
        ):
            f.write(
                f"✅ **SUCCESS**: The optimized LLM assessment tool meets all PRD success criteria.\n\n"
            )
        else:
            f.write(
                f"❌ **PARTIAL SUCCESS**: Some benchmarks not achieved (see details below).\n\n"
            )

        f.write(f"## Experiment Configuration\n\n")
        f.write(f"- **Model**: {df['model_name'].iloc[0]}\n")
        f.write(f"- **Prompt Strategy**: {df['prompt_strategy'].iloc[0]}\n")
        f.write(f"- **Temperature**: {df['temperature'].iloc[0]}\n")
        f.write(
            f"- **Dataset**: Set C ({df['transcript_id'].nunique():,} transcripts)\n"
        )
        f.write(f"- **Total Assessments**: {len(df):,}\n")
        f.write(
            f"- **Success Rate**: {((df['Parsing_Error'].isna().sum() / len(df)) * 100):.1f}%\n\n"
        )

        f.write(f"## Benchmark Performance\n\n")
        f.write(f"### Primary Success Metric (PRD 7.5.3)\n")
        f.write(f"- **Target**: Mean STDEV < {TARGET_STDEV_THRESHOLD}\n")
        f.write(f"- **Achieved**: {benchmark_results['mean_stdev']:.3f}\n")
        f.write(
            f"- **Status**: {'✅ PASSED' if benchmark_results['benchmark_met'] else '❌ FAILED'}\n\n"
        )

        f.write(f"### Coverage Metric\n")
        f.write(
            f"- **Target**: {SUCCESS_PERCENTAGE_TARGET}% of transcripts under threshold\n"
        )
        f.write(f"- **Achieved**: {benchmark_results['percentage_meeting']:.1f}%\n")
        f.write(
            f"- **Status**: {'✅ PASSED' if benchmark_results['percentage_benchmark_met'] else '❌ FAILED'}\n\n"
        )

        f.write(f"## Detailed Statistics\n\n")
        f.write(f"### Consistency Metrics\n")
        f.write(f"- **Mean STDEV**: {consistency_df['stdev_total_score'].mean():.3f}\n")
        f.write(
            f"- **Median STDEV**: {consistency_df['stdev_total_score'].median():.3f}\n"
        )
        f.write(
            f"- **90th Percentile**: {consistency_df['stdev_total_score'].quantile(0.9):.3f}\n"
        )
        f.write(
            f"- **Maximum STDEV**: {consistency_df['stdev_total_score'].max():.3f}\n\n"
        )

        f.write(f"### Score Distribution\n")
        f.write(
            f"- **Mean Score**: {score_stats['total_score_stats']['mean']:.2f} ± {score_stats['total_score_stats']['std']:.2f}\n"
        )
        f.write(
            f"- **Score Range**: {score_stats['total_score_stats']['min']}-{score_stats['total_score_stats']['max']}\n"
        )
        f.write(
            f"- **Median Score**: {score_stats['total_score_stats']['median']:.2f}\n\n"
        )

        f.write(f"## Category-Level Performance\n\n")
        for category, results in benchmark_results["category_results"].items():
            f.write(f"### {category}\n")
            f.write(f"- **Mean STDEV**: {results['mean_stdev']:.3f}\n")
            f.write(
                f"- **Under Threshold**: {results['percentage_under_threshold']:.1f}%\n\n"
            )

        f.write(f"## Data Source Comparison (DoVA vs Nature)\n\n")
        source_data = source_comparison["source_comparison"]
        f.write(f"### Dataset Distribution\n")
        f.write(f"- **DoVA Transcripts**: {source_data['dova_count']:,}\n")
        f.write(f"- **Nature Transcripts**: {source_data['nature_count']:,}\n\n")

        f.write(f"### Consistency Comparison\n")
        f.write(f"- **DoVA Mean STDEV**: {source_data['dova_mean_stdev']:.3f}\n")
        f.write(f"- **Nature Mean STDEV**: {source_data['nature_mean_stdev']:.3f}\n")
        f.write(
            f"- **Statistical Significance**: p = {source_data['stdev_p_value']:.4f}\n"
        )
        f.write(
            f"- **Result**: {'Significant difference' if source_data['stdev_p_value'] < 0.05 else 'No significant difference'}\n\n"
        )

        f.write(f"### Score Comparison\n")
        f.write(f"- **DoVA Mean Score**: {source_data['dova_mean_score']:.2f}\n")
        f.write(f"- **Nature Mean Score**: {source_data['nature_mean_score']:.2f}\n")
        f.write(
            f"- **Score Difference p-value**: {source_data['score_p_value']:.4f}\n\n"
        )

        f.write(f"### Benchmark Achievement\n")
        f.write(
            f"- **DoVA**: {source_data['dova_benchmark_percentage']:.1f}% under threshold\n"
        )
        f.write(
            f"- **Nature**: {source_data['nature_benchmark_percentage']:.1f}% under threshold\n\n"
        )

        f.write(f"## Uncertainty Analysis\n\n")
        f.write(
            f"- **Score-Uncertainty Correlation**: ρ = {uncertainty_results['score_uncertainty_correlation']:.3f}\n"
        )
        f.write(
            f"- **High Uncertainty Threshold**: {uncertainty_results['high_uncertainty_threshold']:.3f}\n"
        )
        f.write(
            f"- **High Uncertainty Cases**: {len(uncertainty_results['high_uncertainty_transcripts'])}\n\n"
        )

        f.write(f"## Conclusions and Recommendations\n\n")
        if (
            benchmark_results["benchmark_met"]
            and benchmark_results["percentage_benchmark_met"]
        ):
            f.write(
                f"The optimized tool configuration (GPT-4o + Few-Shot) successfully meets all PRD success criteria. "
            )
            f.write(
                f"The tool demonstrates excellent consistency with a mean STDEV of {benchmark_results['mean_stdev']:.3f}, "
            )
            f.write(f"well below the target threshold of {TARGET_STDEV_THRESHOLD}. ")
            f.write(
                f"This validates the systematic optimization approach used in Experiments 1-4.\n\n"
            )
            f.write(
                f"**Recommendation**: Proceed with deployment using this configuration.\n"
            )
        else:
            f.write(
                f"While the tool shows promising performance, it does not fully meet all PRD benchmarks. "
            )
            f.write(
                f"Consider additional optimization or adjustment of success criteria based on clinical requirements.\n"
            )

    print(f"✅ Comprehensive report saved: {report_path}")

    # Also save summary statistics as JSON for programmatic access
    summary_data = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": datetime.now().isoformat(),
        "benchmark_results": benchmark_results,
        "score_statistics": score_stats,
        "source_comparison": source_comparison,
        "uncertainty_results": uncertainty_results,
        "dataset_info": {
            "total_assessments": len(df),
            "unique_transcripts": df["transcript_id"].nunique(),
            "success_rate": (df["Parsing_Error"].isna().sum() / len(df)) * 100,
        },
    }

    json_path = os.path.join(
        ANALYSIS_DIR, f"{EXPERIMENT_ID.lower()}_summary_statistics.json"
    )
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)

    print(f"✅ Summary statistics saved: {json_path}")


def main():
    """Main analysis function for Experiment 5."""
    print("🚀 EXPERIMENT 5: FINAL TOOL VALIDATION ANALYSIS")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and validate data
    df = load_and_validate_data()

    # Calculate consistency metrics
    consistency_df = calculate_consistency_metrics(df)

    # Evaluate benchmark performance
    benchmark_results = evaluate_benchmark_performance(consistency_df)

    # Analyze score distributions
    score_stats = analyze_score_distributions(df, consistency_df)

    # Compare data sources (DoVA vs Nature)
    source_comparison = analyze_data_source_differences(df, consistency_df)

    # Explore uncertainty quantification
    uncertainty_results = explore_uncertainty_quantification(df, consistency_df)

    # Create visualizations
    create_visualizations(df, consistency_df, benchmark_results)

    # Save comprehensive report
    save_comprehensive_report(
        df,
        consistency_df,
        benchmark_results,
        score_stats,
        source_comparison,
        uncertainty_results,
    )

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Results saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
