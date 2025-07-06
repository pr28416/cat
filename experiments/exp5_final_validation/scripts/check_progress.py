#!/usr/bin/env python3
"""
Experiment 5: Progress Checker

This utility script checks the current progress of Experiment 5
and provides information about completion status and resume capabilities.
"""

import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp5_final_validation/results/processed_scores"
)
PARTITIONS_FILE = os.path.join(PROJECT_ROOT, "experiments/dataset_partitions.json")

EXPERIMENT_ID = "EXP5_FinalToolValidation"
RESULTS_FILE = f"{EXPERIMENT_ID.lower()}_grading_results.csv"
COMPLETION_FILE = f"{EXPERIMENT_ID.lower()}_grading_results_completion_info.json"


def check_experiment_progress():
    """Check the current progress of Experiment 5."""
    print("=" * 80)
    print("📊 EXPERIMENT 5: PROGRESS CHECK")
    print("=" * 80)
    print(f"📅 Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load expected dataset size
    try:
        with open(PARTITIONS_FILE, "r") as f:
            partitions = json.load(f)
        set_c_size = len(partitions["exp5_set_C"])
        expected_total = set_c_size * 10  # 10 attempts per transcript
    except Exception as e:
        print(f"❌ Error loading dataset partitions: {e}")
        return

    print(f"📋 Expected totals:")
    print(f"   Transcripts: {set_c_size}")
    print(f"   Total assessments: {expected_total}")

    # Check main results file
    results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)

    if not os.path.exists(results_path):
        print(f"\n❌ No results file found: {RESULTS_FILE}")
        print(f"   Status: Experiment not started")
        return

    # Load and analyze results
    try:
        df = pd.read_csv(results_path)

        print(f"\n📊 Current Progress:")
        print(f"   Results file: {RESULTS_FILE}")
        print(f"   Total assessments completed: {len(df):,}")
        print(
            f"   Progress: {len(df)/expected_total*100:.1f}% ({len(df)}/{expected_total})"
        )

        # Analyze by transcript
        transcript_counts = df.groupby("transcript_id").size()
        complete_transcripts = (transcript_counts == 10).sum()
        partial_transcripts = ((transcript_counts > 0) & (transcript_counts < 10)).sum()

        print(f"\n📋 Transcript Analysis:")
        print(f"   Complete transcripts (10/10): {complete_transcripts}")
        print(f"   Partial transcripts: {partial_transcripts}")
        print(f"   Total transcripts touched: {len(transcript_counts)}")
        print(f"   Remaining transcripts: {set_c_size - len(transcript_counts)}")

        # Success rate analysis
        successful = df[df["Parsing_Error"].isna()]
        success_rate = len(successful) / len(df) * 100

        print(f"\n✅ Quality Metrics:")
        print(
            f"   Successful assessments: {len(successful):,}/{len(df):,} ({success_rate:.1f}%)"
        )
        print(f"   Failed assessments: {len(df) - len(successful):,}")

        if len(df) - len(successful) > 0:
            error_types = df[df["Parsing_Error"].notna()][
                "Parsing_Error"
            ].value_counts()
            print(f"   Error breakdown:")
            for error, count in error_types.head(3).items():
                print(f"     {error}: {count}")

        # Time analysis
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_time = df["timestamp"].min()
            latest_time = df["timestamp"].max()
            duration = latest_time - start_time

            print(f"\n⏱️  Timing Analysis:")
            print(f"   Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Latest: {latest_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Duration: {duration}")

            if len(df) > 100:  # Only calculate rate if we have enough data
                rate = len(df) / duration.total_seconds() * 3600  # per hour
                remaining = expected_total - len(df)
                eta_hours = remaining / rate if rate > 0 else float("inf")

                print(f"   Rate: {rate:.1f} assessments/hour")
                if eta_hours < 24:
                    print(f"   ETA: {eta_hours:.1f} hours remaining")
                else:
                    print(f"   ETA: {eta_hours/24:.1f} days remaining")

        # Check for completion info
        completion_path = os.path.join(RESULTS_DIR, COMPLETION_FILE)
        if os.path.exists(completion_path):
            try:
                with open(completion_path, "r") as f:
                    completion_info = json.load(f)

                print(f"\n🎉 EXPERIMENT COMPLETED!")
                print(f"   Completion time: {completion_info['completion_timestamp']}")
                print(f"   Final success rate: {completion_info['success_rate']:.1f}%")
            except Exception as e:
                print(f"   ⚠️  Error reading completion info: {e}")

        # Resumption advice
        if len(df) < expected_total:
            remaining_tasks = expected_total - len(df)
            print(f"\n🔄 RESUMPTION STATUS:")
            print(f"   Remaining tasks: {remaining_tasks:,}")
            print(f"   Resume command: ./run_experiment5.sh")
            print(f"   ✅ Checkpointing active - safe to resume anytime")
        else:
            print(f"\n✅ EXPERIMENT COMPLETE!")
            print(f"   All {expected_total:,} assessments finished")
            print(
                f"   Ready for analysis: python experiments/exp5_final_validation/scripts/02_analyze_final_validation.py"
            )

    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        return

    # Check for backup files
    print(f"\n💾 Backup Files Available:")
    backup_files = []
    for file in os.listdir(RESULTS_DIR):
        if file.startswith(EXPERIMENT_ID.lower()) and (
            "backup" in file or "checkpoint" in file or "safety" in file
        ):
            backup_files.append(file)

    if backup_files:
        print(f"   Found {len(backup_files)} backup/checkpoint files")
        for backup in sorted(backup_files)[-3:]:  # Show last 3
            backup_path = os.path.join(RESULTS_DIR, backup)
            size = os.path.getsize(backup_path) / 1024 / 1024  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(backup_path))
            print(f"     {backup} ({size:.1f}MB, {mtime.strftime('%H:%M:%S')})")
    else:
        print(f"   No backup files found")


def main():
    """Main function to check experiment progress."""
    check_experiment_progress()


if __name__ == "__main__":
    main()
