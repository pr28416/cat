import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import sys
import json

# Add the project root to Python path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

# --- Configuration ---
INPUT_NON_RUBRIC_RESULTS = os.path.join(
    PROJECT_ROOT,
    "experiments/exp1_baseline_utility/results/exp1_non_rubric_grading_results.csv",
)
INPUT_RUBRIC_RESULTS = os.path.join(
    PROJECT_ROOT,
    "experiments/exp1_baseline_utility/results/exp1_rubric_grading_results.csv",
)
INPUT_SYNTHETIC_DATA = os.path.join(
    PROJECT_ROOT, "data/synthetic/exp1_synthetic_transcripts.csv"
)
PROCESSED_SCORES_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp1_baseline_utility/results/processed_scores/"
)
PROCESSED_SCORES_CSV = os.path.join(
    PROCESSED_SCORES_DIR, "exp1_all_grading_attempts.csv"
)
ANALYSIS_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp1_baseline_utility/results/analysis/"
)
ALPHA = 0.05  # Significance level for statistical tests


# --- Helper Functions ---
def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)


def save_plot(fig, filename):
    path = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close(fig)  # Close the figure to free memory


def calculate_accuracy_metrics(df):
    """Calculate accuracy metrics for each grading condition."""
    metrics = {}
    for condition in df["GradingCondition"].unique():
        condition_data = df[df["GradingCondition"] == condition]

        # Calculate mean absolute error
        mae = np.mean(
            np.abs(
                condition_data["Effective_Total_Score"]
                - condition_data["TargetTotalScore_Synthetic"]
            )
        )

        # Calculate root mean squared error
        rmse = np.sqrt(
            np.mean(
                (
                    condition_data["Effective_Total_Score"]
                    - condition_data["TargetTotalScore_Synthetic"]
                )
                ** 2
            )
        )

        # Calculate correlation coefficient
        correlation = condition_data["Effective_Total_Score"].corr(
            condition_data["TargetTotalScore_Synthetic"]
        )

        metrics[condition] = {"MAE": mae, "RMSE": rmse, "Correlation": correlation}

    return metrics


def analyze_score_distribution(df):
    """Analyze the distribution of scores for each grading condition."""
    distribution_stats = {}
    for condition in df["GradingCondition"].unique():
        condition_data = df[df["GradingCondition"] == condition]

        # Calculate basic statistics
        stats = condition_data["Effective_Total_Score"].describe()

        # Calculate skewness and kurtosis
        skewness = stats.skew()
        kurtosis = stats.kurtosis()

        distribution_stats[condition] = {
            "mean": stats["mean"],
            "std": stats["std"],
            "min": stats["min"],
            "max": stats["max"],
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    return distribution_stats


# --- Main Analysis Logic ---
def main():
    print("Starting Experiment 1: Results Analysis...")

    # Create necessary directories
    ensure_dir(PROCESSED_SCORES_DIR)
    ensure_dir(ANALYSIS_OUTPUT_DIR)

    # Load synthetic data first to get target scores
    try:
        df_synthetic = pd.read_csv(INPUT_SYNTHETIC_DATA)
        print(f"Loaded {len(df_synthetic)} synthetic transcripts")
    except FileNotFoundError:
        print(f"Error: Could not find synthetic transcripts at {INPUT_SYNTHETIC_DATA}")
        return

    # Load and combine results
    try:
        df_non_rubric = pd.read_csv(INPUT_NON_RUBRIC_RESULTS)
        df_rubric = pd.read_csv(INPUT_RUBRIC_RESULTS)

        print(f"Loaded {len(df_non_rubric)} non-rubric grading results")
        print(f"Loaded {len(df_rubric)} rubric-based grading results")

        # Rename grading conditions for clarity
        df_non_rubric["GradingCondition"] = "G1_NonRubric"
        df_rubric["GradingCondition"] = "G2_RubricBased"

        # Combine results
        df_attempts = pd.concat([df_non_rubric, df_rubric], ignore_index=True)

        # Merge with synthetic data to get target scores
        df_attempts = df_attempts.merge(
            df_synthetic[["SyntheticTranscriptID", "TargetTotalScore"]],
            left_on="TranscriptID",
            right_on="SyntheticTranscriptID",
            how="left",
        )
        df_attempts.rename(
            columns={"TargetTotalScore": "TargetTotalScore_Synthetic"}, inplace=True
        )

        # Save processed data
        df_attempts.to_csv(PROCESSED_SCORES_CSV, index=False)
        print(
            f"Saved {len(df_attempts)} processed grading attempts to {PROCESSED_SCORES_CSV}"
        )

    except Exception as e:
        print(f"Error processing results: {e}")
        return

    # --- Data Preparation and Cleaning ---
    # Convert relevant columns to numeric, coercing errors to NaN
    score_cols_g1 = ["Total_Score_Reported"]
    score_cols_g2 = [
        "Clarity_of_Language",
        "Lexical_Diversity",
        "Conciseness_and_Completeness",
        "Engagement_with_Health_Information",
        "Health_Literacy_Indicator",
        "Total_Score_Calculated",
        "Total_Score_Reported",
    ]

    for col in score_cols_g1 + score_cols_g2:
        if col in df_attempts.columns:
            df_attempts[col] = pd.to_numeric(df_attempts[col], errors="coerce")

    # Calculate effective total score
    df_attempts["Effective_Total_Score"] = np.nan

    g1_mask = df_attempts["GradingCondition"] == "G1_NonRubric"
    df_attempts.loc[g1_mask, "Effective_Total_Score"] = df_attempts.loc[
        g1_mask, "Total_Score_Reported"
    ]

    g2_mask = df_attempts["GradingCondition"] == "G2_RubricBased"
    df_attempts.loc[
        g2_mask & df_attempts["Total_Score_Calculated"].notna(), "Effective_Total_Score"
    ] = df_attempts.loc[
        g2_mask & df_attempts["Total_Score_Calculated"].notna(),
        "Total_Score_Calculated",
    ]
    df_attempts.loc[
        g2_mask
        & df_attempts["Total_Score_Calculated"].isna()
        & df_attempts["Total_Score_Reported"].notna(),
        "Effective_Total_Score",
    ] = df_attempts.loc[
        g2_mask
        & df_attempts["Total_Score_Calculated"].isna()
        & df_attempts["Total_Score_Reported"].notna(),
        "Total_Score_Reported",
    ]

    # Filter out rows with parsing errors or missing scores
    initial_rows = len(df_attempts)
    df_attempts.dropna(subset=["Effective_Total_Score"], inplace=True)
    df_attempts = df_attempts[
        ~(
            (df_attempts["GradingCondition"] == "G2_RubricBased")
            & df_attempts["Parsing_Error"].notna()
        )
    ]
    print(
        f"Dropped {initial_rows - len(df_attempts)} rows due to parsing errors or missing scores."
    )

    if df_attempts.empty:
        print("No valid data remaining after cleaning. Exiting analysis.")
        return

    # --- 1. Consistency (STDEV) - H1a ---
    print("\n--- H1a: Consistency (STDEV) Analysis ---")
    stdev_results = (
        df_attempts.groupby(["TranscriptID", "GradingCondition"])
        .agg(
            STDEV_Total_Score=("Effective_Total_Score", "std"),
            Mean_Total_Score=(
                "Effective_Total_Score",
                "mean",
            ),  # Keep mean for other uses if needed
            TargetTotalScore_Synthetic=("TargetTotalScore_Synthetic", "first"),
        )
        .reset_index()
    )

    # Ensure STDEV is 0 if only 1 attempt (should not happen with N_ATTEMPTS > 1 but good practice)
    # For N_ATTEMPTS=1, std() is NaN. For N_ATTEMPTS=50 this is not an issue.
    # stdev_results['STDEV_Total_Score'].fillna(0, inplace=True)

    stdev_g1 = stdev_results[stdev_results["GradingCondition"] == "G1_NonRubric"][
        "STDEV_Total_Score"
    ].dropna()
    stdev_g2 = stdev_results[stdev_results["GradingCondition"] == "G2_RubricBased"][
        "STDEV_Total_Score"
    ].dropna()

    summary_stats = []

    if not stdev_g1.empty and not stdev_g2.empty:
        u_stat_stdev, p_value_stdev = stats.mannwhitneyu(
            stdev_g1, stdev_g2, alternative="two-sided"
        )  # PRD asks for comparison, two-sided is appropriate to see if G2 is lower
        summary_stats.append(
            {
                "Metric": "STDEV Total Score (H1a)",
                "Group1_Median": stdev_g1.median(),
                "Group1_IQR": stats.iqr(stdev_g1),
                "Group2_Median": stdev_g2.median(),
                "Group2_IQR": stats.iqr(stdev_g2),
                "U_Statistic": u_stat_stdev,
                "P_Value": p_value_stdev,
                "Significance": (
                    "< 0.001" if p_value_stdev < 0.001 else f"{p_value_stdev:.3f}"
                ),
                "Hypothesis_Supported (G2 lower STDEV)": (
                    "Yes"
                    if p_value_stdev < ALPHA and stdev_g2.median() < stdev_g1.median()
                    else "No"
                ),
            }
        )
        print(f"Mann-Whitney U for STDEV: U={u_stat_stdev}, p={p_value_stdev}")
        print(f"Median STDEV G1: {stdev_g1.median():.3f}, G2: {stdev_g2.median():.3f}")
    else:
        print("Not enough data for STDEV comparison.")
        summary_stats.append(
            {
                "Metric": "STDEV Total Score (H1a)",
                "Error": "Not enough data for comparison",
            }
        )

    # Plot STDEV distributions
    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=stdev_g1,
        label="G1 (Non-Rubric) STDEV",
        kde=True,
        color="blue",
        element="step",
        stat="density",
        common_norm=False,
    )
    sns.histplot(
        data=stdev_g2,
        label="G2 (Rubric-Based) STDEV",
        kde=True,
        color="orange",
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Distribution of STDEVs of Total Scores by Grading Condition (H1a)")
    plt.xlabel("Standard Deviation of Total Score")
    plt.ylabel("Density")
    plt.legend()
    save_plot(plt.gcf(), "exp1_stdev_distribution.png")

    # --- 2. Accuracy (MAE) - H1b ---
    print("\n--- H1b: Accuracy (MAE) Analysis ---")
    # Calculate mean LLM total score per transcript and condition first
    # This 'stdev_results' DataFrame already has Mean_Total_Score and TargetTotalScore_Synthetic

    stdev_results["MAE_MeanScore_vs_TTS"] = np.abs(
        stdev_results["Mean_Total_Score"] - stdev_results["TargetTotalScore_Synthetic"]
    )

    mae_g1 = stdev_results[stdev_results["GradingCondition"] == "G1_NonRubric"][
        "MAE_MeanScore_vs_TTS"
    ].dropna()
    mae_g2 = stdev_results[stdev_results["GradingCondition"] == "G2_RubricBased"][
        "MAE_MeanScore_vs_TTS"
    ].dropna()

    # For Wilcoxon signed-rank test, we need paired data.
    # We should pivot the table to have G1_MAE and G2_MAE as columns for each TranscriptID.
    mae_pivot = stdev_results.pivot_table(
        index="TranscriptID", columns="GradingCondition", values="MAE_MeanScore_vs_TTS"
    ).reset_index()

    # Drop rows where either G1 or G2 MAE is NaN (i.e., transcript not present in both conditions after initial processing)
    mae_pivot.dropna(subset=["G1_NonRubric", "G2_RubricBased"], inplace=True)

    if not mae_pivot.empty and len(mae_pivot) > 0:  # scipy requires N > 0 for wilcoxon
        # Check if G1 and G2 MAE series are identical (can happen if scores are identical)
        # Wilcoxon test throws error if all differences are zero.
        diff = mae_pivot["G1_NonRubric"] - mae_pivot["G2_RubricBased"]
        if np.all(diff == 0):
            print(
                "Wilcoxon test for MAE: All differences are zero. Cannot perform test."
            )
            p_value_mae = 1.0  # Or handle as a special case
            stat_mae = np.nan
        else:
            stat_mae, p_value_mae = stats.wilcoxon(
                mae_pivot["G1_NonRubric"],
                mae_pivot["G2_RubricBased"],
                alternative="two-sided",  # PRD: G2 MAE lower than G1. 'greater' means G1 > G2
            )

        summary_stats.append(
            {
                "Metric": "MAE (Mean Score vs TTS) (H1b)",
                "Group1_Median_MAE": mae_pivot["G1_NonRubric"].median(),
                "Group1_IQR_MAE": stats.iqr(mae_pivot["G1_NonRubric"].dropna()),
                "Group2_Median_MAE": mae_pivot["G2_RubricBased"].median(),
                "Group2_IQR_MAE": stats.iqr(mae_pivot["G2_RubricBased"].dropna()),
                "Wilcoxon_Statistic": stat_mae,
                "P_Value": p_value_mae,
                "Significance": (
                    "< 0.001" if p_value_mae < 0.001 else f"{p_value_mae:.3f}"
                ),
                "Hypothesis_Supported (G2 lower MAE)": (
                    "Yes"
                    if p_value_mae < ALPHA
                    and mae_pivot["G2_RubricBased"].median()
                    < mae_pivot["G1_NonRubric"].median()
                    else "No"
                ),
            }
        )
        print(f"Wilcoxon signed-rank for MAE: W={stat_mae}, p={p_value_mae}")
        print(
            f"Median MAE G1: {mae_pivot['G1_NonRubric'].median():.3f}, G2: {mae_pivot['G2_RubricBased'].median():.3f}"
        )

    else:
        print("Not enough paired data for MAE comparison.")
        summary_stats.append(
            {
                "Metric": "MAE (Mean Score vs TTS) (H1b)",
                "Error": "Not enough paired data for comparison",
            }
        )

    # Plot MAE distributions
    plt.figure(figsize=(10, 6))
    # Need to melt the pivot table back for seaborn histplot with hue
    mae_plot_df = mae_pivot.melt(
        id_vars=["TranscriptID"],
        value_vars=["G1_NonRubric", "G2_RubricBased"],
        var_name="GradingCondition",
        value_name="MAE_MeanScore_vs_TTS",
    )

    sns.histplot(
        data=mae_plot_df,
        x="MAE_MeanScore_vs_TTS",
        hue="GradingCondition",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
    )
    plt.title("Distribution of MAE (Mean Score vs Target) by Grading Condition (H1b)")
    plt.xlabel("Mean Absolute Error (|Mean Score - Target Score|)")
    plt.ylabel("Density")
    plt.legend()
    save_plot(plt.gcf(), "exp1_mae_distribution.png")

    # --- 3. Score Distribution Comparison - H1c ---
    print("\n--- H1c: Score Distribution Analysis ---")
    # This uses the 'stdev_results' df which contains Mean_Total_Score per transcript/condition
    # and TargetTotalScore_Synthetic

    plt.figure(figsize=(12, 7))
    sns.histplot(
        stdev_results[stdev_results["GradingCondition"] == "G1_NonRubric"][
            "Mean_Total_Score"
        ],
        label="G1 (Non-Rubric) Mean Scores",
        kde=True,
        color="blue",
        element="step",
        stat="density",
        common_norm=False,
        bins=16,  # Range 5-20 -> 16 bins for integer scores
    )
    sns.histplot(
        stdev_results[stdev_results["GradingCondition"] == "G2_RubricBased"][
            "Mean_Total_Score"
        ],
        label="G2 (Rubric-Based) Mean Scores",
        kde=True,
        color="orange",
        element="step",
        stat="density",
        common_norm=False,
        bins=16,
    )
    sns.histplot(
        stdev_results[
            "TargetTotalScore_Synthetic"
        ].unique(),  # Use unique target scores for TTS distribution
        label="Synthetic Target Scores (TTS)",
        kde=True,
        color="green",
        element="step",
        stat="density",
        common_norm=False,
        bins=16,
    )
    plt.title("Distribution of Mean LLM Total Scores vs. Synthetic Target Scores (H1c)")
    plt.xlabel("Total Score")
    plt.ylabel("Density")
    plt.legend()
    save_plot(plt.gcf(), "exp1_mean_scores_vs_tts_distribution.png")
    print("Plotted score distributions for G1, G2 (mean scores), and TTS.")
    # Qualitative comparison is done by visual inspection of the plot.
    # KL Divergence could be added here if desired, comparing G1/G2 distributions to TTS.
    # For KL divergence, ensure bins are consistent and represent probabilities.

    # --- Overall Summary Table ---
    df_summary = pd.DataFrame(summary_stats)
    summary_table_path = os.path.join(
        ANALYSIS_OUTPUT_DIR, "exp1_summary_statistics.csv"
    )
    df_summary.to_csv(summary_table_path, index=False)
    print(f"\nSaved summary statistics to: {summary_table_path}")
    print("\nSummary of Statistical Tests:")
    print(df_summary.to_string())

    # --- Remove original accuracy_metrics and distribution_stats calculation ---
    # The new approach integrates these into the hypothesis testing sections.
    # print("\\nAnalyzing score distributions...")
    # distribution_stats = analyze_score_distribution(df_attempts) # Original call
    # print("\\nAnalyzing target vs. actual scores...")
    # accuracy_metrics = calculate_accuracy_metrics(df_attempts) # Original call

    # The helper functions `calculate_accuracy_metrics` and `analyze_score_distribution`
    # are no longer directly used in the main flow as their logic is now part of the
    # hypothesis-driven sections H1a, H1b, H1c for clarity.
    # They can be kept for utility or removed if not needed elsewhere.
    # For now, I will leave them in the file but comment out their calls from main().

    print("\nExperiment 1: Results Analysis script finished.")


if __name__ == "__main__":
    main()
