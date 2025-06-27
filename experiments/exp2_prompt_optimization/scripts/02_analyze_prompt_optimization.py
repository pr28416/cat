import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path to allow importing from other directories
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
from experiments.common.file_utils import ensure_dir_exists, save_plot

# --- Configuration ---
EXPERIMENT_ID = "EXP2_PromptOptimization"
RESULTS_FILE = f"experiments/exp2_prompt_optimization/results/processed_scores/{EXPERIMENT_ID.lower()}_grading_results.csv"
ANALYSIS_OUTPUT_DIR = f"experiments/exp2_prompt_optimization/results/analysis"


def main():
    """
    Main function to analyze the results of the prompt optimization experiment.
    """
    print(f"--- Analyzing Results for {EXPERIMENT_ID} ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"Successfully loaded {len(df)} records from '{RESULTS_FILE}'")
    except FileNotFoundError:
        print(
            f"Error: Results file not found at '{RESULTS_FILE}'. Please run the grading script first."
        )
        return

    # --- Data Quality Check ---
    print("\n--- Data Quality Check ---")
    parsing_errors = df[df["Parsing_Error"].notna()]
    error_counts = parsing_errors.groupby("prompt_name").size()
    total_counts = df.groupby("prompt_name").size()
    error_rate = (error_counts / total_counts * 100).fillna(0)

    quality_df = pd.DataFrame(
        {
            "Total Attempts": total_counts,
            "Parsing Errors": error_counts,
            "Error Rate (%)": error_rate,
        }
    ).fillna(0)

    print("Parsing Error Report by Prompt Strategy:")
    print(quality_df)

    # Filter out rows with parsing errors for STDEV calculation
    df_clean = df[df["Parsing_Error"].isna()].copy()

    if len(df_clean) < len(df):
        print(
            f"\nRemoved {len(df) - len(df_clean)} rows with parsing errors before analysis."
        )

    # Ensure the output directory exists
    ensure_dir_exists(ANALYSIS_OUTPUT_DIR)

    # --- 2a. Arithmetic Reliability Analysis ---
    print("\n--- Arithmetic Reliability Analysis ---")
    arithmetic_errors = df_clean[
        df_clean["Total_Score_Reported"] != df_clean["Total_Score_Calculated"]
    ]
    error_counts_arithmetic = arithmetic_errors.groupby("prompt_name").size()
    total_counts_clean = df_clean.groupby("prompt_name").size()
    arithmetic_error_rate = (error_counts_arithmetic / total_counts_clean * 100).fillna(
        0
    )

    arithmetic_df = pd.DataFrame(
        {
            "Total Clean Attempts": total_counts_clean,
            "Arithmetic Errors": error_counts_arithmetic,
            "Arithmetic Error Rate (%)": arithmetic_error_rate,
        }
    ).fillna(0)

    print("Arithmetic Error Report by Prompt Strategy:")
    print(arithmetic_df)

    # --- 2b. Calculate Standard Deviation for each group (Total Score) ---
    # Group by transcript and prompt strategy, then calculate STDEV of the total score
    stdev_df = (
        df_clean.groupby(["transcript_id", "prompt_name"])["Total_Score_Calculated"]
        .std()
        .reset_index()
    )
    stdev_df.rename(
        columns={"Total_Score_Calculated": "stdev_total_score"}, inplace=True
    )

    print(
        "\nCalculated Standard Deviation of total scores for each transcript and prompt strategy (on clean data):"
    )
    print(stdev_df.head())

    # --- 3. Statistical Analysis (as per PRD) ---
    print("\n--- Statistical Analysis ---")

    # Pivot the data to have prompt strategies as columns
    stdev_pivot = stdev_df.pivot(
        index="transcript_id", columns="prompt_name", values="stdev_total_score"
    )

    # Extract the STDEV values for each prompt strategy
    zero_shot_stdevs = stdev_pivot["zero_shot"].dropna()
    few_shot_stdevs = stdev_pivot["few_shot"].dropna()
    cot_stdevs = stdev_pivot["cot"].dropna()

    # Friedman Test
    # This test is used to detect differences in treatments across multiple test attempts (paired data).
    # It's a non-parametric alternative to the repeated measures ANOVA.
    friedman_stat, p_value = stats.friedmanchisquare(
        zero_shot_stdevs, few_shot_stdevs, cot_stdevs
    )

    print(f"Friedman Test Results:")
    print(f"  - Statistic: {friedman_stat:.4f}")
    print(f"  - P-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(
            "The Friedman test is significant. There are differences between the prompt strategies."
        )

        # Post-hoc analysis (Wilcoxon signed-rank test for paired samples with Bonferroni correction)
        print(
            "\nPerforming post-hoc Wilcoxon signed-rank tests with Bonferroni correction..."
        )
        comparisons = [
            ("zero_shot", "few_shot"),
            ("zero_shot", "cot"),
            ("few_shot", "cot"),
        ]
        p_values_posthoc = {}

        for comp in comparisons:
            stat, p = stats.wilcoxon(stdev_pivot[comp[0]], stdev_pivot[comp[1]])
            p_values_posthoc[f"{comp[0]}_vs_{comp[1]}"] = p

        # Bonferroni correction
        corrected_alpha = alpha / len(comparisons)
        print(f"Bonferroni corrected alpha: {corrected_alpha:.4f}")

        for comp, p in p_values_posthoc.items():
            verdict = "Significant" if p < corrected_alpha else "Not Significant"
            print(f"  - {comp}: p-value = {p:.4f} ({verdict})")

    else:
        print(
            "The Friedman test is not significant. No statistical difference found between prompt strategies for total score consistency."
        )

    # --- 4. Category-Level Consistency Analysis ---
    print("\n--- Category-Level Consistency Analysis ---")

    # Melt the dataframe to analyze categories
    categories = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
    ]
    df_melted = df_clean.melt(
        id_vars=["transcript_id", "prompt_name"],
        value_vars=categories,
        var_name="category",
        value_name="score",
    )

    # Calculate STDEV for each category
    category_stdev_df = (
        df_melted.groupby(["transcript_id", "prompt_name", "category"])["score"]
        .std()
        .reset_index()
    )
    category_stdev_df.rename(columns={"score": "stdev_score"}, inplace=True)

    # Get the mean STDEV for each prompt/category combo
    mean_category_stdev = (
        category_stdev_df.groupby(["prompt_name", "category"])["stdev_score"]
        .mean()
        .reset_index()
    )

    print("Mean Standard Deviation by Category and Prompt:")
    print(
        mean_category_stdev.pivot(
            index="category", columns="prompt_name", values="stdev_score"
        )
    )

    # --- 5. Visualization ---
    print("\nGenerating visualizations...")

    # Boxplot of STDEVs by Prompt Strategy (Total Score)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=stdev_df, x="prompt_name", y="stdev_total_score", ax=ax)
    sns.stripplot(
        data=stdev_df,
        x="prompt_name",
        y="stdev_total_score",
        ax=ax,
        color=".25",
        size=4,
    )

    ax.set_title("Distribution of Score Standard Deviation by Prompt Strategy")
    ax.set_xlabel("Prompt Strategy")
    ax.set_ylabel("Standard Deviation of Total Score (per transcript)")
    save_plot(fig, "exp2_stdev_distribution_boxplot.png", ANALYSIS_OUTPUT_DIR)

    # Barplot of STDEVs by Category and Prompt
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(
        data=mean_category_stdev,
        x="category",
        y="stdev_score",
        hue="prompt_name",
        ax=ax,
    )

    ax.set_title("Mean Score Standard Deviation by Rubric Category and Prompt")
    ax.set_xlabel("Rubric Category")
    ax.set_ylabel("Mean Standard Deviation of Score")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, "exp2_category_stdev_by_prompt.png", ANALYSIS_OUTPUT_DIR)

    # Score Distribution Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(
        data=df_clean,
        x="Total_Score_Calculated",
        hue="prompt_name",
        fill=True,
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Distribution of Scores by Prompt Strategy")
    ax.set_xlabel("Total Score (Calculated)")
    ax.set_ylabel("Density")
    save_plot(fig, "exp2_score_distribution_by_prompt.png", ANALYSIS_OUTPUT_DIR)

    # --- 6. Determine Winning Strategy ---
    mean_stdevs = (
        stdev_df.groupby("prompt_name")["stdev_total_score"].mean().sort_values()
    )
    print("\n--- Conclusion ---")
    print("Mean Standard Deviation per Prompt Strategy:")
    print(mean_stdevs)

    winner = mean_stdevs.index[0]
    print(f"\nWinning Prompt Strategy (lowest mean STDEV): **{winner.upper()}**")

    print(f"\n--- Analysis Complete. Outputs saved to '{ANALYSIS_OUTPUT_DIR}' ---")


if __name__ == "__main__":
    main()
