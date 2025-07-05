# This script will run the grading for Experiment 3: Model Comparison

import os
import sys
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

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

# --- Constants for Test Run ---
EXPERIMENT_ID = "EXP3_ModelComparison"
MODELS_TO_TEST = [
    "gpt-4o-2024-08-06",  # Best general GPT-4o model
    "gpt-4o-mini-2024-07-18",  # Best cost-efficient model
    "gpt-4.1-2025-04-14",  # Best GPT-4.1 model
    "o3-2025-04-16",  # Best o3 reasoning model
    "o3-mini-2025-01-31",  # o3 mini for comparison
]
TEMPERATURE = 0.1
NUM_TRANSCRIPTS_TO_PROCESS = 40  # Curated experiment
NUM_ATTEMPTS_PER_MODEL = 20  # Full experiment

# --- File Paths ---
# Get project root directory (4 levels up from this script)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PARTITIONS_FILE = os.path.join(PROJECT_ROOT, "experiments/dataset_partitions.json")
TRANSCRIPT_DIR = os.path.join(PROJECT_ROOT, "data/cleaned")
PROMPT_FILE = os.path.join(
    PROJECT_ROOT,
    "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt",
)  # Winning prompt from Exp2
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/processed_scores"
)
RUBRIC_FILE = os.path.join(PROJECT_ROOT, "experiments/common/rubric_v5.md")


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
    start_time = time.time()
    raw_response, error = None, None
    try:
        # o1/o3/o4 models require temperature=1 and no max_tokens limit
        if model.startswith(("o1-", "o3-", "o4-")):
            raw_response = generate_text(
                prompt, model=model, temperature=1.0, max_tokens=None
            )
        else:
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
        # All prompts in Exp3 use the rubric-based few-shot prompt
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "transcript_id": transcript_id,
        "model_name": model,
        "attempt_num": attempt_num,
        "temperature": TEMPERATURE,
        "raw_response": raw_response,
        "latency": latency,
        **parsed_scores,
    }


def main():
    """
    Main function to run the model comparison experiment.
    """
    print(f"--- Starting Experiment: {EXPERIMENT_ID} ---")
    print(
        f"Running test with {NUM_TRANSCRIPTS_TO_PROCESS} transcript(s), {NUM_ATTEMPTS_PER_MODEL} attempt(s) for {len(MODELS_TO_TEST)} model(s)."
    )

    # --- 1. Load Data and Prompt ---
    print("Loading data partitions and winning prompt...")
    with open(DATA_PARTITIONS_FILE, "r") as f:
        partitions = json.load(f)

    # Use Set B for this experiment as per PRD
    set_b_transcripts = partitions["exp3_set_B"]

    # Slice for test run
    transcripts_to_process = set_b_transcripts[:NUM_TRANSCRIPTS_TO_PROCESS]

    # Load the winning prompt and the rubric
    prompt_template = load_text_file(PROMPT_FILE)
    rubric = load_text_file(RUBRIC_FILE)
    print(f"  - Loaded prompt: {os.path.basename(PROMPT_FILE)}")
    print(f"  - Loaded rubric: {os.path.basename(RUBRIC_FILE)}")

    # --- 2. Create All Grading Tasks ---
    tasks = []
    for transcript_id in transcripts_to_process:
        for model in MODELS_TO_TEST:
            for i in range(1, NUM_ATTEMPTS_PER_MODEL + 1):
                tasks.append(
                    {
                        "transcript_id": transcript_id,
                        "model": model,
                        "prompt_template": prompt_template,
                        "rubric_text": rubric,
                        "attempt_num": i,
                    }
                )

    total_tasks = len(tasks)
    print(f"\nCreated {total_tasks} grading tasks to execute.")

    # --- 3. Check for Existing Results (Checkpointing) ---
    output_filename = os.path.join(
        RESULTS_DIR, f"{EXPERIMENT_ID.lower()}_grading_results_curated_models.csv"
    )

    existing_results = []
    completed_tasks = set()

    if os.path.exists(output_filename):
        print(f"\nFound existing results file: {os.path.basename(output_filename)}")
        try:
            existing_df = pd.read_csv(output_filename)
            existing_results = existing_df.to_dict("records")

            # Create set of completed task identifiers
            for result in existing_results:
                task_id = f"{result['transcript_id']}_{result['model_name']}_{result['attempt_num']}"
                completed_tasks.add(task_id)

            print(f"Found {len(existing_results)} existing results.")
            print(f"Resuming from checkpoint...")
        except Exception as e:
            print(f"Error reading existing results: {e}")
            print("Starting fresh...")
            existing_results = []
            completed_tasks = set()
    else:
        print(f"\nNo existing results found. Starting fresh...")

    # Filter out already completed tasks
    remaining_tasks = []
    for task in tasks:
        task_id = f"{task['transcript_id']}_{task['model']}_{task['attempt_num']}"
        if task_id not in completed_tasks:
            remaining_tasks.append(task)

    print(f"Tasks remaining: {len(remaining_tasks)} out of {total_tasks}")

    if len(remaining_tasks) == 0:
        print("All tasks already completed!")
        return

    # --- 4. Run Grading in Parallel ---
    results = existing_results.copy()  # Start with existing results
    # Using 12 workers since we have 6 API keys (2 workers per key)
    max_workers = 3  # Conservative setting for 1 API key
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(run_single_grading_attempt, task): task
            for task in remaining_tasks
        }

        completed_count = 0
        for future in tqdm(
            as_completed(future_to_task),
            total=len(remaining_tasks),
            desc="Grading transcripts",
        ):
            result = future.result()
            results.append(result)
            completed_count += 1

            # Save progress every 100 completions
            if completed_count % 100 == 0:
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(output_filename, index=False)
                print(f"\nCheckpoint saved after {completed_count} completions...")

    # --- 5. Save Results ---
    if not results:
        print("\nNo results to save. Exiting.")
        return

    print("\nGrading complete. Saving results...")
    results_df = pd.DataFrame(results)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(output_filename, index=False)

    print(f"--- Results saved to {output_filename} ---")
    print(f"--- Experiment {EXPERIMENT_ID} Finished ---")


if __name__ == "__main__":
    main()
