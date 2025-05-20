import os
import sys
import pandas as pd
import json
import time
import multiprocessing as mp
from functools import partial
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to Python path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from ai.openai import generate_text_with_messages
from experiments.common.enums import GradingCondition
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.file_utils import ensure_dir_exists

# --- Configuration ---
INPUT_SYNTHETIC_DATA_PATH = os.path.join(
    PROJECT_ROOT, "data/synthetic/exp1_synthetic_transcripts.csv"
)
OUTPUT_RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp1_baseline_utility/results/"
)
OUTPUT_NON_RUBRIC_RESULTS_PATH = os.path.join(
    OUTPUT_RESULTS_DIR, "exp1_non_rubric_grading_results.csv"
)
OUTPUT_RUBRIC_RESULTS_PATH = os.path.join(
    OUTPUT_RESULTS_DIR, "exp1_rubric_grading_results.csv"
)

NON_RUBRIC_PROMPT_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments/exp1_baseline_utility/prompts/p1_1_non_rubric_grading_prompt.txt",
)
RUBRIC_PROMPT_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments/exp1_baseline_utility/prompts/p1_2_rubric_based_grading_prompt.txt",
)
RUBRIC_PATH = os.path.join(PROJECT_ROOT, "experiments/common/rubric_v5.md")

# Per PRD section 6 - Default parameters
LLM_EVALUATION_MODEL = "gpt-4o-mini"  # Updated to use available model
TEMPERATURE_EVALUATION = 0.1
MAX_TOKENS_EVALUATION = 500

# Number of times to grade each transcript for reliability
NUM_ATTEMPTS_PER_TRANSCRIPT = 50

# Number of processes for parallel processing
NUM_PROCESSES = max(1, mp.cpu_count() - 1)  # Leave one CPU free


# --- Helper Functions ---
def load_text_file(filepath: str) -> str:
    """Loads text content from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def grade_transcript(
    args: Tuple[str, str, GradingCondition, str, float, int, str, int, Optional[str]],
) -> Dict[str, Any]:
    """
    Grade a single transcript using the specified approach.
    Args are unpacked from a tuple to support multiprocessing.
    """
    (
        transcript_text,
        prompt_template,
        grading_condition,
        model,
        temperature,
        max_tokens,
        transcript_id,
        attempt_num,
        rubric_text,
    ) = args

    # Prepare the prompt
    prompt = prompt_template.replace("{{TranscriptText}}", transcript_text)
    if rubric_text and "{{RubricV5Text}}" in prompt:
        prompt = prompt.replace("{{RubricV5Text}}", rubric_text)

    # Prepare messages format for the API call
    messages = [
        {
            "role": "system",
            "content": "You are an expert health communication assessor. Grade the transcript according to the instructions.",
        },
        {"role": "user", "content": prompt},
    ]

    # Call the LLM API for grading
    print(
        f"Grading transcript {transcript_id} (Attempt {attempt_num}) with {grading_condition.value} approach"
    )
    try:
        response = generate_text_with_messages(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Parse the scores from the LLM response
        parsed_results = parse_scores_from_response(
            response_text=response,
            grading_condition=grading_condition,
            transcript_id=transcript_id,
            attempt_num=attempt_num,
        )

        # Add metadata to the results
        result = {
            "TranscriptID": transcript_id,
            "AttemptNum": attempt_num,
            "GradingCondition": grading_condition.value,
            "RawLLMResponse": response,
            "LLMModel": model,
            "Temperature": temperature,
            "MaxTokens": max_tokens,
        }

        # Add the parsed scores to the results
        result.update(parsed_results)

        return result

    except Exception as e:
        print(f"Error grading transcript {transcript_id}: {e}")
        # Return a result with error information
        return {
            "TranscriptID": transcript_id,
            "AttemptNum": attempt_num,
            "GradingCondition": grading_condition.value,
            "RawLLMResponse": str(e),
            "LLMModel": model,
            "Temperature": temperature,
            "MaxTokens": max_tokens,
            "Parsing_Error": f"API Error: {str(e)}",
        }


def main():
    print("Starting Experiment 1: Synthetic Transcript Grading...")
    print(f"Using {NUM_PROCESSES} processes for parallel grading")

    # Ensure output directory exists
    ensure_dir_exists(OUTPUT_RESULTS_DIR)

    # Load prompt templates and rubric
    try:
        non_rubric_prompt = load_text_file(NON_RUBRIC_PROMPT_PATH)
        rubric_prompt = load_text_file(RUBRIC_PROMPT_PATH)
        rubric_text = load_text_file(RUBRIC_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not load required file: {e}. Exiting.")
        return

    # Load synthetic transcripts
    try:
        df_synthetic = pd.read_csv(INPUT_SYNTHETIC_DATA_PATH)
        print(f"Loaded {len(df_synthetic)} synthetic transcripts")
    except FileNotFoundError:
        print(
            f"Error: Could not find synthetic transcripts at {INPUT_SYNTHETIC_DATA_PATH}"
        )
        return
    except Exception as e:
        print(f"Error loading synthetic transcripts: {e}")
        return

    # Prepare grading tasks for both conditions
    non_rubric_tasks = []
    rubric_tasks = []

    for _, row in df_synthetic.iterrows():
        transcript_id = row["SyntheticTranscriptID"]
        transcript_text = row["GeneratedTranscript"]

        # Add non-rubric grading tasks
        for attempt in range(NUM_ATTEMPTS_PER_TRANSCRIPT):
            non_rubric_tasks.append(
                (
                    transcript_text,
                    non_rubric_prompt,
                    GradingCondition.NON_RUBRIC,
                    LLM_EVALUATION_MODEL,
                    TEMPERATURE_EVALUATION,
                    MAX_TOKENS_EVALUATION,
                    transcript_id,
                    attempt + 1,
                    None,
                )
            )

        # Add rubric-based grading tasks
        for attempt in range(NUM_ATTEMPTS_PER_TRANSCRIPT):
            rubric_tasks.append(
                (
                    transcript_text,
                    rubric_prompt,
                    GradingCondition.RUBRIC_BASED,
                    LLM_EVALUATION_MODEL,
                    TEMPERATURE_EVALUATION,
                    MAX_TOKENS_EVALUATION,
                    transcript_id,
                    attempt + 1,
                    rubric_text,
                )
            )

    # Process non-rubric grading tasks in parallel
    print("\nStarting non-rubric grading...")
    with mp.Pool(NUM_PROCESSES) as pool:
        non_rubric_results = pool.map(grade_transcript, non_rubric_tasks)

    # Process rubric-based grading tasks in parallel
    print("\nStarting rubric-based grading...")
    with mp.Pool(NUM_PROCESSES) as pool:
        rubric_results = pool.map(grade_transcript, rubric_tasks)

    # Save results
    if non_rubric_results:
        df_non_rubric = pd.DataFrame(non_rubric_results)
        df_non_rubric.to_csv(OUTPUT_NON_RUBRIC_RESULTS_PATH, index=False)
        print(f"Saved non-rubric results to {OUTPUT_NON_RUBRIC_RESULTS_PATH}")

    if rubric_results:
        df_rubric = pd.DataFrame(rubric_results)
        df_rubric.to_csv(OUTPUT_RUBRIC_RESULTS_PATH, index=False)
        print(f"Saved rubric-based results to {OUTPUT_RUBRIC_RESULTS_PATH}")

    print("\nExperiment 1: Synthetic Transcript Grading script finished.")


if __name__ == "__main__":
    main()
