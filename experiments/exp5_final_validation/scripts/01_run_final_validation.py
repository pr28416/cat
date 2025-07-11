#!/usr/bin/env python3
"""
Experiment 5: Final Tool Validation

This script runs the final validation of the optimized LLM assessment tool configuration
on Set C (577 unseen transcripts) using the optimal model and prompt strategy identified
from previous experiments.

Configuration:
- Model: gpt-4o-2024-08-06 (optimal balance of consistency and reasoning quality)
- Prompt: Few-Shot strategy (winner from Experiment 2)
- Dataset: Set C (577 transcripts, unseen in previous experiments)
- Attempts: 10 per transcript (as per PRD)
- Total: 5,770 grading attempts
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from ai.openai import generate_text
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition
import openai

# --- Experiment Configuration (as per PRD and previous results) ---
EXPERIMENT_ID = "EXP5_FinalToolValidation"
OPTIMIZED_MODEL = "gpt-4o-2024-08-06"  # Winner from Exp 3 + Exp 4 analysis
TEMPERATURE = 0.1
NUM_TRANSCRIPTS_TO_USE = 250  # Reduced scope for deeper analysis
NUM_ATTEMPTS_PER_TRANSCRIPT = 20  # Increased attempts per transcript
MAX_WORKERS = 4  # Parallel processing (user requested max 4 threads)

# API Key Management
MAX_429_ERRORS_PER_KEY = 10  # Disable key after this many 429 errors
api_key_error_counts = {}  # Track 429 errors per key
disabled_api_keys = set()  # Track disabled keys

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PARTITIONS_FILE = os.path.join(PROJECT_ROOT, "experiments/dataset_partitions.json")
TRANSCRIPT_DIR = os.path.join(PROJECT_ROOT, "data/cleaned")
PROMPT_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt",
)  # Winning Few-Shot prompt from Exp2
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp5_final_validation/results/processed_scores"
)
RUBRIC_FILE = os.path.join(PROJECT_ROOT, "experiments/common/rubric_v5.md")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_single_grading_attempt(attempt_data):
    """
    Runs a single grading attempt for a given transcript using the optimized tool configuration.

    Args:
        attempt_data (dict): Contains transcript_id, prompt_template, rubric_text, attempt_num

    Returns:
        dict: Results including scores, timing, and any errors
    """
    transcript_id = attempt_data["transcript_id"]
    prompt_template = attempt_data["prompt_template"]
    rubric_text = attempt_data["rubric_text"]
    attempt_num = attempt_data["attempt_num"]

    try:
        # Load transcript
        transcript_text = load_text_file(os.path.join(TRANSCRIPT_DIR, transcript_id))

        # Format the prompt with rubric and transcript
        prompt = prompt_template.replace("{{rubric}}", rubric_text).replace(
            "{{transcript}}", transcript_text
        )

        # Get LLM completion with timing
        start_time = time.time()
        try:
            raw_response = generate_text(
                prompt, model=OPTIMIZED_MODEL, temperature=TEMPERATURE, max_tokens=None
            )
            latency = time.time() - start_time
        except openai.RateLimitError as e:
            # Handle rate limit errors specifically
            latency = time.time() - start_time
            raise Exception(f"Rate limit error (429): {str(e)}")
        except Exception as e:
            # Handle other API errors
            latency = time.time() - start_time
            raise Exception(f"API error: {str(e)}")

        # Parse the response (Few-Shot prompt uses rubric-based format)
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )

        return {
            "experiment_id": EXPERIMENT_ID,
            "transcript_id": transcript_id,
            "model_name": OPTIMIZED_MODEL,
            "prompt_strategy": "few_shot",
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "latency": latency,
            **parsed_scores,
        }

    except Exception as e:
        # Handle any errors gracefully
        return {
            "experiment_id": EXPERIMENT_ID,
            "transcript_id": transcript_id,
            "model_name": OPTIMIZED_MODEL,
            "prompt_strategy": "few_shot",
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": None,
            "latency": None,
            "Parsing_Error": f"Error: {str(e)}",
        }


def load_existing_results(output_filename):
    """
    Load existing results for checkpointing/resuming capability.

    Args:
        output_filename (str): Path to results file

    Returns:
        tuple: (existing_results_list, completed_tasks_set)
    """
    existing_results = []
    completed_tasks = set()

    if os.path.exists(output_filename):
        print(f"\n📂 Found existing results file: {os.path.basename(output_filename)}")
        try:
            existing_df = pd.read_csv(output_filename)
            existing_results = existing_df.to_dict("records")

            # Create set of completed task identifiers for fast lookup
            for result in existing_results:
                task_id = f"{result['transcript_id']}_{result['attempt_num']}"
                completed_tasks.add(task_id)

            print(f"   ✅ Loaded {len(existing_results)} existing results")
            print(f"   🔄 Resuming from checkpoint...")
        except Exception as e:
            print(f"   ❌ Error reading existing results: {e}")
            print(f"   🆕 Starting fresh...")
            existing_results = []
            completed_tasks = set()
    else:
        print(f"\n🆕 No existing results found. Starting fresh experiment...")

    return existing_results, completed_tasks


def create_task_list(transcripts, prompt_template, rubric_text, completed_tasks):
    """
    Create list of grading tasks, filtering out already completed ones.

    Args:
        transcripts (list): List of transcript IDs
        prompt_template (str): Formatted prompt template
        rubric_text (str): Rubric content
        completed_tasks (set): Set of completed task IDs

    Returns:
        list: List of task dictionaries
    """
    tasks = []
    for transcript_id in transcripts:
        for attempt_num in range(1, NUM_ATTEMPTS_PER_TRANSCRIPT + 1):
            task_id = f"{transcript_id}_{attempt_num}"

            if task_id not in completed_tasks:
                tasks.append(
                    {
                        "transcript_id": transcript_id,
                        "prompt_template": prompt_template,
                        "rubric_text": rubric_text,
                        "attempt_num": attempt_num,
                    }
                )

    return tasks


def save_results_incrementally(results, output_filename, batch_size=50):
    """
    Save results incrementally to prevent data loss.

    Args:
        results (list): List of result dictionaries
        output_filename (str): Path to save results
        batch_size (int): How often to save (every N results)
    """
    if len(results) % batch_size == 0 and len(results) > 0:
        try:
            # Create backup of existing file before overwriting
            if os.path.exists(output_filename):
                backup_filename = output_filename.replace(
                    ".csv", f'_backup_{datetime.now().strftime("%H%M%S")}.csv'
                )
                import shutil

                shutil.copy2(output_filename, backup_filename)

            # Save current results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_filename, index=False)

            # Also save a timestamped checkpoint
            checkpoint_filename = output_filename.replace(
                ".csv", f'_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            results_df.to_csv(checkpoint_filename, index=False)

            print(
                f"   💾 Checkpoint saved: {len(results)} results ({datetime.now().strftime('%H:%M:%S')})"
            )

        except Exception as e:
            print(f"   ⚠️  Warning: Failed to save checkpoint: {e}")
            # Continue execution even if checkpoint fails


def main():
    """
    Main function to run Experiment 5: Final Tool Validation.
    """
    print("=" * 80)
    print(f"🚀 STARTING EXPERIMENT 5: FINAL TOOL VALIDATION")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🤖 Optimized Model: {OPTIMIZED_MODEL}")
    print(f"📝 Prompt Strategy: Few-Shot (winner from Experiment 2)")
    print(f"🌡️  Temperature: {TEMPERATURE}")
    print(f"📊 Transcripts to process: {NUM_TRANSCRIPTS_TO_USE}")
    print(f"🔄 Attempts per transcript: {NUM_ATTEMPTS_PER_TRANSCRIPT}")
    print(
        f"🎯 Total target assessments: {NUM_TRANSCRIPTS_TO_USE * NUM_ATTEMPTS_PER_TRANSCRIPT}"
    )

    # --- 1. Load Configuration and Data ---
    print(f"\n📚 Loading experiment configuration...")

    # Load dataset partitions
    with open(DATA_PARTITIONS_FILE, "r") as f:
        partitions = json.load(f)

    set_c_transcripts = partitions["exp5_set_C"]
    print(f"   📊 Set C transcripts available: {len(set_c_transcripts)}")

    # Use only first N transcripts for deeper analysis
    set_c_transcripts = set_c_transcripts[:NUM_TRANSCRIPTS_TO_USE]
    print(f"   🎯 Using first {NUM_TRANSCRIPTS_TO_USE} transcripts for deeper analysis")

    # Load prompt template and rubric
    prompt_template = load_text_file(PROMPT_FILE)
    rubric_text = load_text_file(RUBRIC_FILE)
    print(f"   📝 Loaded Few-Shot prompt: {os.path.basename(PROMPT_FILE)}")
    print(f"   📋 Loaded rubric: {os.path.basename(RUBRIC_FILE)}")

    # --- 2. Setup Checkpointing ---
    output_filename = os.path.join(
        RESULTS_DIR, f"{EXPERIMENT_ID.lower()}_grading_results.csv"
    )

    existing_results, completed_tasks = load_existing_results(output_filename)

    # --- 3. Create Task List ---
    print(f"\n🎯 Creating task list...")
    tasks = create_task_list(
        set_c_transcripts, prompt_template, rubric_text, completed_tasks
    )

    total_expected = len(set_c_transcripts) * NUM_ATTEMPTS_PER_TRANSCRIPT
    print(f"   📋 Total expected tasks: {total_expected}")
    print(f"   ✅ Already completed: {len(completed_tasks)}")
    print(f"   🔄 Remaining tasks: {len(tasks)}")

    if len(tasks) == 0:
        print(f"\n🎉 All tasks already completed! Experiment finished.")
        return

    # --- 4. Run Grading with Progress Tracking ---
    print(f"\n⚡ Starting parallel grading with {MAX_WORKERS} workers...")
    print(f"   🎯 Target: {len(tasks)} remaining assessments")

    results = existing_results.copy()  # Start with existing results

    # Track rate limit errors for early termination
    rate_limit_error_count = 0
    max_rate_limit_errors = 50  # Stop after this many 429 errors

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(run_single_grading_attempt, task): task for task in tasks
        }

        # Process completed tasks with progress bar
        completed_count = 0
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="🔄 Processing transcripts",
            unit="assessments",
        ):
            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                # Incremental saving for data safety (every 50 results)
                save_results_incrementally(results, output_filename)

                # Additional safety save every 500 results
                if completed_count % 500 == 0:
                    safety_filename = output_filename.replace(
                        ".csv", f"_safety_{completed_count}.csv"
                    )
                    pd.DataFrame(results).to_csv(safety_filename, index=False)
                    print(f"   🛡️  Safety backup created: {safety_filename}")

            except Exception as e:
                task = future_to_task[future]
                error_msg = str(e)

                # Check if this is a rate limit error
                if "Rate limit error (429)" in error_msg:
                    rate_limit_error_count += 1
                    print(
                        f"\n⚠️  Rate limit error #{rate_limit_error_count} for {task['transcript_id']}: {e}"
                    )

                    # Check if we should stop early
                    if rate_limit_error_count >= max_rate_limit_errors:
                        print(
                            f"\n🛑 EARLY TERMINATION: Too many rate limit errors ({rate_limit_error_count})"
                        )
                        print(
                            f"   Stopping experiment to prevent further API key exhaustion"
                        )
                        break
                else:
                    print(f"\n❌ Error processing {task['transcript_id']}: {e}")

                # Log the error but continue processing
                error_result = {
                    "experiment_id": EXPERIMENT_ID,
                    "transcript_id": task["transcript_id"],
                    "model_name": OPTIMIZED_MODEL,
                    "prompt_strategy": "few_shot",
                    "attempt_num": task["attempt_num"],
                    "temperature": TEMPERATURE,
                    "timestamp": datetime.now().isoformat(),
                    "raw_response": None,
                    "latency": None,
                    "Parsing_Error": f"Processing Error: {str(e)}",
                }
                results.append(error_result)

    # --- 5. Final Save and Summary ---
    print(f"\n💾 Saving final results...")

    # Create final backup before saving
    if os.path.exists(output_filename):
        final_backup = output_filename.replace(
            ".csv", f'_final_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        import shutil

        shutil.copy2(output_filename, final_backup)
        print(f"   🛡️  Final backup created: {final_backup}")

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_filename, index=False)

    # Also save a completion timestamp file
    completion_info = {
        "experiment_id": EXPERIMENT_ID,
        "completion_timestamp": datetime.now().isoformat(),
        "total_results": len(results_df),
        "total_transcripts": (
            results_df["transcript_id"].nunique() if len(results_df) > 0 else 0
        ),
        "success_rate": (
            len(results_df[results_df["Parsing_Error"].isna()]) / len(results_df) * 100
            if len(results_df) > 0
            else 0
        ),
    }

    completion_file = output_filename.replace(".csv", "_completion_info.json")
    with open(completion_file, "w") as f:
        json.dump(completion_info, f, indent=2)

    print(f"   ✅ Completion info saved: {completion_file}")

    # Calculate success rate
    successful_results = results_df[results_df["Parsing_Error"].isna()]
    success_rate = len(successful_results) / len(results_df) * 100

    print(f"\n🎉 EXPERIMENT 5 COMPLETED!")
    print("=" * 50)
    print(f"📊 Final Statistics:")
    print(f"   Total attempts: {len(results_df)}")
    print(f"   Successful: {len(successful_results)} ({success_rate:.1f}%)")
    print(f"   Transcripts processed: {results_df['transcript_id'].nunique()}")
    print(
        f"   Average attempts per transcript: {len(results_df) / results_df['transcript_id'].nunique():.1f}"
    )
    print(f"   Rate limit errors: {rate_limit_error_count}")
    print(f"   Results saved to: {output_filename}")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if rate_limit_error_count >= max_rate_limit_errors:
        print(f"⚠️  Experiment terminated early due to rate limiting")
        print(f"   Consider using different API keys or reducing concurrency")
    elif success_rate < 95:
        print(f"⚠️  Warning: Success rate below 95%. Review errors before analysis.")
    else:
        print(f"✅ High success rate achieved. Ready for analysis!")


if __name__ == "__main__":
    main()
