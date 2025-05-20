import os
import sys
import pandas as pd
import glob
import time
from typing import Dict, Any, List, Optional

# Add the project root to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from ai.openai import generate_text_with_messages
from experiments.common.enums import GradingCondition
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.file_utils import ensure_dir_exists, load_text_file

# --- Configuration ---
INPUT_REAL_DATA_DIR = "../../../data/cleaned/"
OUTPUT_RESULTS_DIR = "../results/"
OUTPUT_NON_RUBRIC_RESULTS_PATH = os.path.join(
    OUTPUT_RESULTS_DIR, "exp2_non_rubric_grading_results.csv"
)
OUTPUT_RUBRIC_RESULTS_PATH = os.path.join(
    OUTPUT_RESULTS_DIR, "exp2_rubric_grading_results.csv"
)

NON_RUBRIC_PROMPT_PATH = "../prompts/p2_1_non_rubric_grading_prompt.txt"
RUBRIC_PROMPT_PATH = "../prompts/p2_2_rubric_based_grading_prompt.txt"
RUBRIC_PATH = "../../../experiments/common/rubric_v5.md"

# Per PRD section 6 - Default parameters
LLM_EVALUATION_MODEL = "gpt-4.1-mini"  # Mini/smaller model for scoring
TEMPERATURE_EVALUATION = 0.3
MAX_TOKENS_EVALUATION = 500

# Number of times to grade each transcript for reliability
NUM_ATTEMPTS_PER_TRANSCRIPT = 3


def grade_transcript(
    transcript_text: str,
    prompt_template: str,
    grading_condition: GradingCondition,
    model: str,
    temperature: float,
    max_tokens: int,
    transcript_id: str,
    attempt_num: int,
    rubric_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Grade a single transcript using the specified approach.

    Returns a dictionary with the grading results and metadata.
    """
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


def load_transcript_files():
    """Load transcript files from the cleaned data directory."""
    transcript_files = glob.glob(os.path.join(INPUT_REAL_DATA_DIR, "*.txt"))
    transcripts = []

    for file_path in transcript_files:
        transcript_id = os.path.basename(file_path).replace(".txt", "")
        transcript_text = load_text_file(file_path)
        transcripts.append(
            {"TranscriptID": transcript_id, "TranscriptText": transcript_text}
        )

    return transcripts


def main():
    print("Starting Experiment 2: Real Transcript Grading...")

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

    # Load real transcripts
    real_transcripts = load_transcript_files()
    print(f"Loaded {len(real_transcripts)} real transcripts from {INPUT_REAL_DATA_DIR}")

    # Initialize results containers
    non_rubric_results = []
    rubric_results = []

    # Process each transcript
    for transcript in real_transcripts:
        transcript_id = transcript["TranscriptID"]
        transcript_text = transcript["TranscriptText"]

        print(f"\nProcessing transcript {transcript_id}")

        # Run multiple attempts for each grading condition
        for attempt in range(1, NUM_ATTEMPTS_PER_TRANSCRIPT + 1):
            # 1. Non-rubric grading
            non_rubric_result = grade_transcript(
                transcript_text=transcript_text,
                prompt_template=non_rubric_prompt,
                grading_condition=GradingCondition.NON_RUBRIC,
                model=LLM_EVALUATION_MODEL,
                temperature=TEMPERATURE_EVALUATION,
                max_tokens=MAX_TOKENS_EVALUATION,
                transcript_id=transcript_id,
                attempt_num=attempt,
            )
            non_rubric_results.append(non_rubric_result)

            # Brief pause to avoid rate limits
            time.sleep(0.5)

            # 2. Rubric-based grading
            rubric_result = grade_transcript(
                transcript_text=transcript_text,
                prompt_template=rubric_prompt,
                grading_condition=GradingCondition.RUBRIC_BASED,
                model=LLM_EVALUATION_MODEL,
                temperature=TEMPERATURE_EVALUATION,
                max_tokens=MAX_TOKENS_EVALUATION,
                transcript_id=transcript_id,
                attempt_num=attempt,
                rubric_text=rubric_text,
            )
            rubric_results.append(rubric_result)

            # Pause between API calls to avoid rate limits
            time.sleep(0.5)

    # Save results to CSV
    if non_rubric_results:
        df_non_rubric = pd.DataFrame(non_rubric_results)
        df_non_rubric.to_csv(OUTPUT_NON_RUBRIC_RESULTS_PATH, index=False)
        print(
            f"Saved {len(non_rubric_results)} non-rubric grading results to {OUTPUT_NON_RUBRIC_RESULTS_PATH}"
        )

    if rubric_results:
        df_rubric = pd.DataFrame(rubric_results)
        df_rubric.to_csv(OUTPUT_RUBRIC_RESULTS_PATH, index=False)
        print(
            f"Saved {len(rubric_results)} rubric-based grading results to {OUTPUT_RUBRIC_RESULTS_PATH}"
        )

    print("\nExperiment 2: Real Transcript Grading completed.")


if __name__ == "__main__":
    main()
