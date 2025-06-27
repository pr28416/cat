import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

# Add project root to path to allow importing from other directories
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

from ai.openai import generate_text
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition

# --- Constants from PRD ---
EXPERIMENT_ID = "EXP2_PromptOptimization"
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
NUM_ATTEMPTS = 25  # Number of grading attempts per transcript per prompt

# --- File Paths ---
DATA_PARTITIONS_FILE = "experiments/dataset_partitions.json"
TRANSCRIPT_DIR = "data/cleaned"
PROMPT_DIR = "experiments/exp2_prompt_optimization/prompts"
RESULTS_DIR = "experiments/exp2_prompt_optimization/results/processed_scores"
RUBRIC_FILE = "experiments/common/rubric.md"

PROMPT_TEMPLATES = {
    "zero_shot": "p2_1_zero_shot_prompt.txt",
    "few_shot": "p2_2_few_shot_prompt.txt",
    "cot": "p2_3_cot_prompt.txt",
}


def run_single_grading_attempt(attempt_data):
    """
    Runs a single grading attempt for a given transcript and prompt.
    """
    transcript_id = attempt_data["transcript_id"]
    prompt_name = attempt_data["prompt_name"]
    prompt_template = attempt_data["prompt_template"]
    rubric_text = attempt_data["rubric_text"]
    attempt_num = attempt_data["attempt_num"]

    transcript_text = load_text_file(os.path.join(TRANSCRIPT_DIR, transcript_id))

    # Format the prompt
    prompt = prompt_template.replace("{{rubric}}", rubric_text).replace(
        "{{transcript}}", transcript_text
    )

    # Get LLM completion
    start_time = time.time()
    raw_response, error = None, None
    try:
        raw_response = generate_text(prompt, model=LLM_MODEL, temperature=TEMPERATURE)
    except Exception as e:
        error = f"API Error: {e}"
    latency = time.time() - start_time

    # Parse the response
    if error:
        parsed_scores = {"Parsing_Error": error}
    else:
        # All prompts in Exp2 are rubric-based
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )

    return {
        "experiment_id": EXPERIMENT_ID,
        "transcript_id": transcript_id,
        "prompt_name": prompt_name,
        "attempt_num": attempt_num,
        "llm_model": LLM_MODEL,
        "temperature": TEMPERATURE,
        "raw_response": raw_response,
        "latency": latency,
        **parsed_scores,
    }


def main():
    """
    Main function to run the prompt optimization experiment.
    """
    print(f"--- Starting Experiment: {EXPERIMENT_ID} ---")

    # --- 1. Load Data and Prompts ---
    print("Loading data partitions, rubric, and prompt templates...")
    with open(DATA_PARTITIONS_FILE, "r") as f:
        partitions = json.load(f)

    set_a_transcripts = partitions["exp2_set_A"]
    rubric = load_text_file(RUBRIC_FILE)

    prompts = {}
    for name, filename in PROMPT_TEMPLATES.items():
        prompts[name] = load_text_file(os.path.join(PROMPT_DIR, filename))
        print(f"  - Loaded prompt: {name}")

    # --- 2. Create All Grading Tasks ---
    tasks = []
    for transcript_id in set_a_transcripts:
        for prompt_name, prompt_template in prompts.items():
            for i in range(1, NUM_ATTEMPTS + 1):
                tasks.append(
                    {
                        "transcript_id": transcript_id,
                        "prompt_name": prompt_name,
                        "prompt_template": prompt_template,
                        "rubric_text": rubric,
                        "attempt_num": i,
                    }
                )

    total_tasks = len(tasks)
    print(f"\nCreated {total_tasks} grading tasks to execute.")
    print(
        f"({len(set_a_transcripts)} transcripts x {len(prompts)} prompts x {NUM_ATTEMPTS} attempts)"
    )

    # --- 3. Run Grading in Parallel ---
    results = []
    with ThreadPoolExecutor(max_workers=25) as executor:
        # Using tqdm to show progress
        future_to_task = {
            executor.submit(run_single_grading_attempt, task): task for task in tasks
        }
        for future in tqdm(
            as_completed(future_to_task), total=total_tasks, desc="Grading transcripts"
        ):
            result = future.result()
            results.append(result)

    # --- 4. Save Results ---
    print("\nGrading complete. Saving results...")
    results_df = pd.DataFrame(results)

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    output_filename = os.path.join(
        RESULTS_DIR, f"{EXPERIMENT_ID.lower()}_grading_results.csv"
    )
    results_df.to_csv(output_filename, index=False)

    print(f"--- Results saved to {output_filename} ---")
    print(f"--- Experiment {EXPERIMENT_ID} Finished ---")


if __name__ == "__main__":
    main()
