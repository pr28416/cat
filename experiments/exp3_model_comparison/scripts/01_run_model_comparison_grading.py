#!/usr/bin/env python3
"""
Experiment 3: Model Comparison — GPT-5 Extension Run

Adds gpt-5-nano and gpt-5-mini results to the existing curated_models CSV
that already contains legacy model data (gpt-4o, gpt-4o-mini, gpt-4.1, o3, o3-mini).

Configuration (matching the original experiment):
- Models: gpt-5-nano, gpt-5-mini
- Transcripts: 40  (Set B from dataset_partitions.json)
- Attempts per model: 20
- Temperature: 0.1
- Total new grading attempts: 1,600  (40 × 20 × 2)
"""

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(PROJECT_ROOT)

# Load .env so OPENAI_API_KEY is available for ai/openai.py
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from ai.openai import generate_text
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition

# --- Configuration ---
EXPERIMENT_ID = "EXP3_ModelComparison"
MODELS_TO_TEST = [
    "gpt-5-nano",
    "gpt-5-mini",
]
TEMPERATURE = 0.1
NUM_TRANSCRIPTS_TO_PROCESS = 40
NUM_ATTEMPTS_PER_MODEL = 20
MAX_WORKERS = 7  # Conservative for a single API key

# --- File Paths ---
DATA_PARTITIONS_FILE = os.path.join(PROJECT_ROOT, "experiments/dataset_partitions.json")
TRANSCRIPT_DIR = os.path.join(PROJECT_ROOT, "data/cleaned")
PROMPT_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt",
)
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/processed_scores"
)
RUBRIC_FILE = os.path.join(PROJECT_ROOT, "experiments/common/rubric_v5.md")
OUTPUT_FILE = os.path.join(
    RESULTS_DIR, f"{EXPERIMENT_ID.lower()}_grading_results_curated_models.csv"
)


def run_single_grading_attempt(attempt_data):
    """
    Runs a single grading attempt for a given transcript and model.
    """
    transcript_id = attempt_data["transcript_id"]
    model = attempt_data["model"]
    prompt_template = attempt_data["prompt_template"]
    rubric_text = attempt_data["rubric_text"]
    attempt_num = attempt_data["attempt_num"]

    transcript_text = load_text_file(os.path.join(TRANSCRIPT_DIR, transcript_id))

    # Format the prompt with both the rubric and the transcript
    prompt = prompt_template.replace("{{rubric}}", rubric_text).replace(
        "{{transcript}}", transcript_text
    )

    # Get LLM completion
    # GPT-5 models only support temperature=1 (handled by ai/openai.py)
    actual_temperature = 1.0 if model.startswith("gpt-5") else TEMPERATURE
    start_time = time.time()
    raw_response, error = None, None
    try:
        raw_response = generate_text(
            prompt, model=model, temperature=TEMPERATURE, max_tokens=None
        )
    except Exception as e:
        error = f"API Error: {e}"
    latency = time.time() - start_time

    # Parse the response
    if error:
        parsed_scores = {"Parsing_Error": error}
    else:
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "transcript_id": transcript_id,
        "model_name": model,
        "attempt_num": attempt_num,
        "temperature": actual_temperature,
        "raw_response": raw_response,
        "latency": latency,
        **parsed_scores,
    }


def main():
    print(f"--- Experiment 3: GPT-5 Extension Run ---")
    print(f"Models: {MODELS_TO_TEST}")
    print(
        f"Config: {NUM_TRANSCRIPTS_TO_PROCESS} transcripts × "
        f"{NUM_ATTEMPTS_PER_MODEL} attempts × {len(MODELS_TO_TEST)} models = "
        f"{NUM_TRANSCRIPTS_TO_PROCESS * NUM_ATTEMPTS_PER_MODEL * len(MODELS_TO_TEST)} tasks"
    )

    # --- 1. Load Data and Prompt ---
    print("\nLoading data partitions and prompt...")
    with open(DATA_PARTITIONS_FILE, "r") as f:
        partitions = json.load(f)

    set_b_transcripts = partitions["exp3_set_B"]
    transcripts_to_process = set_b_transcripts[:NUM_TRANSCRIPTS_TO_PROCESS]

    prompt_template = load_text_file(PROMPT_FILE)
    rubric = load_text_file(RUBRIC_FILE)
    print(f"  Prompt: {os.path.basename(PROMPT_FILE)}")
    print(f"  Rubric: {os.path.basename(RUBRIC_FILE)}")
    print(f"  Transcripts: {len(transcripts_to_process)}")

    # --- 2. Load existing results for checkpointing ---
    existing_results = []
    completed_tasks = set()

    if os.path.exists(OUTPUT_FILE):
        print(f"\nFound existing results: {os.path.basename(OUTPUT_FILE)}")
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_results = existing_df.to_dict("records")

            for result in existing_results:
                task_id = f"{result['transcript_id']}_{result['model_name']}_{result['attempt_num']}"
                completed_tasks.add(task_id)

            print(f"  {len(existing_results)} existing results loaded")

            # Show which models already have data
            existing_models = existing_df["model_name"].unique()
            print(f"  Models with data: {sorted(existing_models)}")
        except Exception as e:
            print(f"  Error reading existing results: {e}")
            print("  Will append without checkpoint awareness")
            existing_results = []
            completed_tasks = set()
    else:
        print(f"\nNo existing results file found. Starting fresh...")

    # --- 3. Create tasks (only for GPT-5 models, skip already-completed) ---
    tasks = []
    for transcript_id in transcripts_to_process:
        for model in MODELS_TO_TEST:
            for attempt_num in range(1, NUM_ATTEMPTS_PER_MODEL + 1):
                task_id = f"{transcript_id}_{model}_{attempt_num}"
                if task_id not in completed_tasks:
                    tasks.append(
                        {
                            "transcript_id": transcript_id,
                            "model": model,
                            "prompt_template": prompt_template,
                            "rubric_text": rubric,
                            "attempt_num": attempt_num,
                        }
                    )

    print(f"\nTasks to run: {len(tasks)}")
    if len(tasks) == 0:
        print("All GPT-5 tasks already completed!")
        return

    # --- 4. Run Grading in Parallel ---
    results = existing_results.copy()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {
            executor.submit(run_single_grading_attempt, task): task for task in tasks
        }

        completed_count = 0
        for future in tqdm(
            as_completed(future_to_task),
            total=len(tasks),
            desc="Grading (GPT-5 models)",
        ):
            result = future.result()
            results.append(result)
            completed_count += 1

            # Save checkpoint every 100 completions
            if completed_count % 100 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(OUTPUT_FILE, index=False)
                print(f"\n  Checkpoint saved ({completed_count}/{len(tasks)} new)...")

    # --- 5. Save Final Results ---
    print("\nSaving final results...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    new_count = len(results) - len(existing_results)
    print(f"\nDone! Added {new_count} new results.")
    print(f"Total results in file: {len(results)}")
    print(f"Output: {OUTPUT_FILE}")

    # Per-model summary
    for model in MODELS_TO_TEST:
        model_results = [r for r in results if r.get("model_name") == model]
        errors = [r for r in model_results if r.get("Parsing_Error")]
        print(f"  {model}: {len(model_results)} results ({len(errors)} errors)")


if __name__ == "__main__":
    main()
