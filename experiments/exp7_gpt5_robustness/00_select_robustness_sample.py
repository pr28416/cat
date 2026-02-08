"""
Sample Selection for GPT-5 Robustness Check

This script selects 50 transcripts from Exp5 for GPT-5 reliability assessment.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# Configuration
EXP5_RESULTS = os.path.join(PROJECT_ROOT, "experiments/exp5_final_validation/results/processed_scores")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/exp7_gpt5_robustness/")
SAMPLE_SIZE = 50


def select_robustness_sample():
    """Select stratified sample for GPT-5 robustness check."""
    
    print("=" * 80)
    print("SELECTING SAMPLE FOR GPT-5 ROBUSTNESS CHECK")
    print("=" * 80)
    print()
    
    # Find and load most recent exp5 results
    import glob
    files = glob.glob(os.path.join(EXP5_RESULTS, "exp5_finaltoolvalidation_grading_results_checkpoint*.csv"))
    if not files:
        raise FileNotFoundError("No exp5 results found")
    
    latest_file = sorted(files)[-1]
    print(f"Loading data from: {os.path.basename(latest_file)}")
    
    df = pd.read_csv(latest_file)
    df = df.dropna(subset=["Total_Score_Calculated"])
    
    # Compute per-transcript statistics
    transcript_stats = df.groupby("transcript_id").agg({
        "Total_Score_Calculated": ["mean", "std", "count"]
    }).reset_index()
    
    transcript_stats.columns = ["transcript_id", "mean_score", "stdev", "n_attempts"]
    
    # Add data source
    transcript_stats["source"] = transcript_stats["transcript_id"].apply(
        lambda x: "DoVA" if x.startswith("DOVA_") else "Nature"
    )
    
    # Add score category (tertiles)
    tertiles = transcript_stats["mean_score"].quantile([0.33, 0.67])
    transcript_stats["score_category"] = pd.cut(
        transcript_stats["mean_score"],
        bins=[0, tertiles[0.33], tertiles[0.67], 21],
        labels=["Low", "Medium", "High"]
    )
    
    # Add consistency category
    transcript_stats["consistency_category"] = pd.cut(
        transcript_stats["stdev"],
        bins=[0, 0.25, np.inf],
        labels=["High Consistency", "Moderate Consistency"]
    )
    
    print(f"\nTotal transcripts: {len(transcript_stats)}")
    print(f"\nScore distribution:")
    print(transcript_stats.groupby("score_category").size())
    print(f"\nSource distribution:")
    print(transcript_stats.groupby("source").size())
    print(f"\nConsistency distribution:")
    print(transcript_stats.groupby("consistency_category").size())
    
    # Stratified sampling
    # Goal: Balanced across source, score category, and include both consistency levels
    
    selected = []
    
    for source in ["DoVA", "Nature"]:
        for score_cat in ["Low", "Medium", "High"]:
            subset = transcript_stats[
                (transcript_stats["source"] == source) &
                (transcript_stats["score_category"] == score_cat)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Sample 4-5 from each combination
            n_sample = min(5, len(subset), max(4, SAMPLE_SIZE // 12))
            
            # Randomly sample to get mix of consistency levels
            if len(subset) >= n_sample:
                sample = subset.sample(n=n_sample, random_state=42)
            else:
                sample = subset
            
            selected.append(sample)
    
    selected_df = pd.concat(selected, ignore_index=True)
    
    # Adjust to exact sample size if needed
    if len(selected_df) > SAMPLE_SIZE:
        selected_df = selected_df.sample(n=SAMPLE_SIZE, random_state=42)
    elif len(selected_df) < SAMPLE_SIZE:
        # Add more randomly from remaining
        remaining = transcript_stats[~transcript_stats["transcript_id"].isin(selected_df["transcript_id"])]
        additional = remaining.sample(n=SAMPLE_SIZE - len(selected_df), random_state=42)
        selected_df = pd.concat([selected_df, additional], ignore_index=True)
    
    print(f"\n✓ Selected {len(selected_df)} transcripts")
    print(f"\nFinal distribution by score category:")
    print(selected_df.groupby("score_category").size())
    print(f"\nFinal distribution by source:")
    print(selected_df.groupby("source").size())
    print(f"\nFinal distribution by consistency:")
    print(selected_df.groupby("consistency_category").size())
    
    print(f"\nScore statistics:")
    print(f"  Mean: {selected_df['mean_score'].mean():.2f}")
    print(f"  Range: [{selected_df['mean_score'].min():.1f}, {selected_df['mean_score'].max():.1f}]")
    
    print(f"\nConsistency (STDEV) statistics:")
    print(f"  Mean: {selected_df['stdev'].mean():.3f}")
    print(f"  Median: {selected_df['stdev'].median():.3f}")
    
    # Save sample
    output_file = os.path.join(OUTPUT_DIR, "selected_transcripts_for_gpt5_robustness.csv")
    selected_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to: {output_file}")
    
    return selected_df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    select_robustness_sample()

