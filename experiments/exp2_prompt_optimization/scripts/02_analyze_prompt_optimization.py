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

    # Ensure the output directory exists
    ensure_dir_exists(ANALYSIS_OUTPUT_DIR)

    # --- 2. Calculate Standard Deviation for each group ---
    # Group by transcript and prompt strategy, then calculate STDEV of the total score
    stdev_df = (
        df.groupby(["transcript_id", "prompt_name"])["Total_Score_Reported"]
        .std()
        .reset_index()
    )
    stdev_df.rename(columns={"Total_Score_Reported": "stdev_total_score"}, inplace=True)

    print(
        "\nCalculated Standard Deviation of total scores for each transcript and prompt strategy:"
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
            "The Friedman test is not significant. No statistical difference found between prompt strategies."
        )

    # --- 4. Visualization ---
    print("\nGenerating visualizations...")

    # Boxplot of STDEVs by Prompt Strategy
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

    # --- 5. Determine Winning Strategy ---
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
