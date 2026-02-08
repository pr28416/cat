"""
Sample Selection for Human Rater Pilot

This script selects 30-40 transcripts stratified by CHAT score distribution
and data source for the human validation pilot study.
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/exp6_human_rater_pilot/")
TARGET_SAMPLE_SIZE = 35  # Target 35 transcripts

def select_sample():
    """Select stratified sample of transcripts for human rating."""
    
    print("=" * 80)
    print("SELECTING SAMPLE FOR HUMAN RATER PILOT")
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
    df = df.dropna(subset=["Total_Score_Calculated"])  # Remove failed parses
    
    # Compute per-transcript statistics
    transcript_stats = df.groupby("transcript_id").agg({
        "Total_Score_Calculated": ["mean", "std", "count"]
    }).reset_index()
    
    transcript_stats.columns = ["transcript_id", "mean_score", "stdev", "n_attempts"]
    
    # Add data source
    transcript_stats["source"] = transcript_stats["transcript_id"].apply(
        lambda x: "DoVA" if x.startswith("DOVA_") else "Nature"
    )
    
    # Add score category
    transcript_stats["score_category"] = pd.cut(
        transcript_stats["mean_score"],
        bins=[0, 12, 15, 21],
        labels=["Low (5-12)", "Medium (13-15)", "High (16-20)"]
    )
    
    print(f"\nTotal transcripts with scores: {len(transcript_stats)}")
    print(f"\nScore distribution:")
    print(transcript_stats.groupby("score_category").size())
    print(f"\nSource distribution:")
    print(transcript_stats.groupby("source").size())
    
    # Stratified sampling strategy:
    # 1. Prioritize high consistency (low STDEV) for reliable baseline
    # 2. Balance across score categories
    # 3. Include both data sources
    
    selected = []
    
    # Sample from each combination of score_category and source
    for score_cat in ["Low (5-12)", "Medium (13-15)", "High (16-20)"]:
        for source in ["DoVA", "Nature"]:
            subset = transcript_stats[
                (transcript_stats["score_category"] == score_cat) &
                (transcript_stats["source"] == source)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # Sort by consistency (low STDEV first), then randomly sample
            subset = subset.sort_values("stdev")
            
            # Sample size proportional to category, aim for ~6 per combination
            n_sample = min(6, len(subset))
            
            # Take most consistent transcripts from this stratum
            sample = subset.head(n_sample)
            selected.append(sample)
    
    selected_df = pd.concat(selected, ignore_index=True)
    
    # If we have too many, prioritize most consistent overall
    if len(selected_df) > TARGET_SAMPLE_SIZE:
        selected_df = selected_df.sort_values("stdev").head(TARGET_SAMPLE_SIZE)
    
    print(f"\n✓ Selected {len(selected_df)} transcripts")
    print(f"\nFinal distribution by score category:")
    print(selected_df.groupby("score_category").size())
    print(f"\nFinal distribution by source:")
    print(selected_df.groupby("source").size())
    print(f"\nConsistency (STDEV) statistics:")
    print(f"  Mean: {selected_df['stdev'].mean():.3f}")
    print(f"  Median: {selected_df['stdev'].median():.3f}")
    print(f"  Max: {selected_df['stdev'].max():.3f}")
    
    # Save sample
    output_file = os.path.join(OUTPUT_DIR, "selected_transcripts_for_rating.csv")
    selected_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Create rating sheet template
    rating_template = selected_df[["transcript_id", "source", "mean_score"]].copy()
    rating_template["rater_id"] = ""
    rating_template["engagement_score"] = ""
    rating_template["health_literacy_score"] = ""
    rating_template["comments"] = ""
    
    template_file = os.path.join(OUTPUT_DIR, "rating_sheet_template.csv")
    rating_template.to_csv(template_file, index=False)
    
    print(f"✓ Created rating template: {template_file}")
    
    return selected_df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    select_sample()

