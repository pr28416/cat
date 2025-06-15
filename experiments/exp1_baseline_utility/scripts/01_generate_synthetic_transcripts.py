import os
import random
import sys
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
import json
from typing import List, Dict

# Add the project root to Python path for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)
from ai.openai import generate_text
from experiments.common.parsing_utils import extract_transcript_from_response

# --- Configuration ---
NUM_SYNTHETIC_TRANSCRIPTS = 50
LLM_GENERATION_MODEL = "gpt-4o-mini"
TEMPERATURE_GENERATION = 0.7
MAX_TOKENS_GENERATION = 1000
OUTPUT_DIR_SYNTHETIC_DATA = os.path.join(PROJECT_ROOT, "data/synthetic/")
PROMPT_TEMPLATE_PATH = os.path.join(
    PROJECT_ROOT,
    "experiments/exp1_baseline_utility/prompts/synthetic_transcript_generation_prompt.txt",
)
RUBRIC_PATH = os.path.join(PROJECT_ROOT, "experiments/common/rubric_v5.md")
OUTPUT_CSV_PATH = os.path.join(
    OUTPUT_DIR_SYNTHETIC_DATA, "exp1_synthetic_transcripts.csv"
)
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR_SYNTHETIC_DATA, "generation_checkpoint.json")
NUM_PROCESSES = max(1, mp.cpu_count() - 1)

# Define target score distribution
SCORE_DISTRIBUTION = {
    "low": (5, 10),  # 20% of transcripts
    "medium": (11, 15),  # 40% of transcripts
    "high": (16, 20),  # 40% of transcripts
}
RUBRIC_CATEGORIES = [
    "Clarity_of_Language",
    "Lexical_Diversity",
    "Conciseness_and_Completeness",
    "Engagement_with_Health_Information",
    "Health_Literacy_Indicator",
]


# --- Helper Functions ---
def load_text_file(filepath):
    """Loads text content from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_checkpoint():
    """Load the generation checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"completed_indices": [], "results": []}


def save_checkpoint(completed_indices, results):
    """Save the current generation progress."""
    checkpoint_data = {"completed_indices": completed_indices, "results": results}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def _partition_score(
    total_score: int, num_parts: int = 5, min_val: int = 1, max_val: int = 4
) -> List[int]:
    """Partitions a total score into a list of integers with constraints."""
    if not (num_parts * min_val <= total_score <= num_parts * max_val):
        # This can happen for edge cases like a total score of 5 (only one partition: 1,1,1,1,1)
        # or 20 (only 4,4,4,4,4). If it's a valid score, we can find a partition.
        # A more robust check for validity before calling might be useful, but for 5-20 it's always possible.
        # For now, we'll rely on valid inputs.
        pass

    parts = [min_val] * num_parts
    remainder = total_score - sum(parts)

    while remainder > 0:
        # Find indices that can be incremented
        eligible_indices = [i for i, part in enumerate(parts) if part < max_val]
        if not eligible_indices:
            # Should not happen if total_score is valid
            break
        # Choose a random part to increment
        idx_to_increment = random.choice(eligible_indices)
        parts[idx_to_increment] += 1
        remainder -= 1

    random.shuffle(parts)
    return parts


def generate_target_scores(num_transcripts: int) -> List[Dict[str, int]]:
    """Generate target subscores and total scores following the defined distribution."""
    target_scores_list = []
    num_low = int(num_transcripts * 0.2)
    num_medium = int(num_transcripts * 0.4)
    num_high = num_transcripts - num_low - num_medium

    # Generate total scores first
    total_scores = []
    total_scores.extend(
        [random.randint(*SCORE_DISTRIBUTION["low"]) for _ in range(num_low)]
    )
    total_scores.extend(
        [random.randint(*SCORE_DISTRIBUTION["medium"]) for _ in range(num_medium)]
    )
    total_scores.extend(
        [random.randint(*SCORE_DISTRIBUTION["high"]) for _ in range(num_high)]
    )
    random.shuffle(total_scores)

    # For each total score, generate a valid partition of subscores
    for total_score in total_scores:
        subscores = _partition_score(total_score, len(RUBRIC_CATEGORIES))
        score_dict = dict(zip(RUBRIC_CATEGORIES, subscores))
        score_dict["TargetTotalScore"] = total_score
        target_scores_list.append(score_dict)

    return target_scores_list


def call_llm_api(prompt, model, temperature, max_tokens):
    """Call the OpenAI API to generate a synthetic transcript."""
    try:
        print(
            f"Calling OpenAI API - Model: {model}, Temp: {temperature}, Max Tokens: {max_tokens}"
        )
        response = generate_text(
            prompt=prompt, model=model, temperature=temperature, max_tokens=max_tokens
        )
        return response
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise


def generate_single_transcript(args):
    """Generate a single synthetic transcript based on target subscores."""
    i, generation_prompt_template, rubric_text, target_scores_dict = args
    synthetic_transcript_id = f"SYNTH_EXP1_{i+1:03d}"
    target_total_score = target_scores_dict["TargetTotalScore"]

    print(
        f"\nGenerating transcript {i+1}/{NUM_SYNTHETIC_TRANSCRIPTS} (ID: {synthetic_transcript_id}, Target Score: {target_total_score})"
    )
    print(f"  Target Subscores: {target_scores_dict}")

    # Construct the prompt with explicit quality level and subscores
    quality_level = (
        "low"
        if target_total_score <= 10
        else "medium" if target_total_score <= 15 else "high"
    )
    prompt = generation_prompt_template.replace("{{TTS}}", str(target_total_score))
    prompt = prompt.replace("{{RubricV5Text}}", rubric_text)
    prompt = prompt.replace("{{QualityLevel}}", quality_level)

    # Replace subscore placeholders
    for category, score in target_scores_dict.items():
        if category != "TargetTotalScore":
            prompt = prompt.replace(f"{{{{Target_{category}}}}}", str(score))

    # Call LLM API
    try:
        llm_response = call_llm_api(
            prompt=prompt,
            model=LLM_GENERATION_MODEL,
            temperature=TEMPERATURE_GENERATION,
            max_tokens=MAX_TOKENS_GENERATION,
        )

        # Extract transcript from response
        final_transcript_text = extract_transcript_from_response(llm_response)

        if final_transcript_text:
            # The result now includes the target subscores
            result = {
                "SyntheticTranscriptID": synthetic_transcript_id,
                "GeneratedTranscript": final_transcript_text,
                "FullLLMResponse": llm_response,
            }
            # Add all target scores (subscores and total) to the result dictionary
            for category, score in target_scores_dict.items():
                # Prepending "Target_" to make column names clear
                result[f"Target_{category.replace('Target', '')}"] = score

            print(
                f"Successfully generated and processed transcript {synthetic_transcript_id}"
            )
            return result
        else:
            print(
                f"Failed to extract transcript for {synthetic_transcript_id}. Skipping."
            )
            return None

    except Exception as e:
        print(f"Error during LLM call or processing for {synthetic_transcript_id}: {e}")
        return None


# --- Main Script Logic ---
def main():
    print("Starting Experiment 1: Synthetic Transcript Generation...")
    print(f"Using {NUM_PROCESSES} processes for parallel generation")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR_SYNTHETIC_DATA, exist_ok=True)

    # Load prompt template and rubric
    try:
        generation_prompt_template = load_text_file(PROMPT_TEMPLATE_PATH)
        rubric_text = load_text_file(RUBRIC_PATH)
    except FileNotFoundError as e:
        print(f"Error: Could not load required file: {e}. Exiting.")
        return

    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    completed_indices = checkpoint["completed_indices"]
    results = checkpoint["results"]

    # Generate target scores for ALL transcripts (not just remaining ones)
    # This ensures consistent score distribution regardless of checkpoints
    all_target_scores = generate_target_scores(NUM_SYNTHETIC_TRANSCRIPTS)

    # Calculate remaining transcripts
    remaining_count = NUM_SYNTHETIC_TRANSCRIPTS - len(completed_indices)

    # Get target scores for remaining transcripts only
    target_scores_to_generate = [
        all_target_scores[i]
        for i in range(NUM_SYNTHETIC_TRANSCRIPTS)
        if i not in completed_indices
    ]

    # Prepare arguments for parallel processing
    args = [
        (
            len(completed_indices) + i,
            generation_prompt_template,
            rubric_text,
            target_scores_to_generate[i],
        )
        for i in range(len(target_scores_to_generate))
    ]

    if not args:
        print("All transcripts have been generated. Loading existing results...")
    else:
        print(f"Generating {len(args)} remaining transcripts...")
        # Generate transcripts in parallel
        with mp.Pool(NUM_PROCESSES) as pool:
            new_results = pool.map(generate_single_transcript, args)

        # Filter out None results and update checkpoint
        valid_new_results = [r for r in new_results if r is not None]
        results.extend(valid_new_results)

        # Update completed indices correctly
        new_completed_indices = [
            len(completed_indices) + i
            for i, r in enumerate(new_results)
            if r is not None
        ]
        completed_indices.extend(new_completed_indices)
        save_checkpoint(completed_indices, results)

    if results:
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Ensure all required columns are present
        required_columns = [
            "SyntheticTranscriptID",
            "GeneratedTranscript",
            "FullLLMResponse",
            "Target_TotalScore",  # Note: name changed slightly to be consistent
        ] + [f"Target_{cat}" for cat in RUBRIC_CATEGORIES]

        df = df.reindex(columns=required_columns + ["QualityLevel"])

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in results: {missing_columns}")
            for col in missing_columns:
                df[col] = None

        # Add QualityLevel column based on Target_TotalScore for convenience
        df["QualityLevel"] = df["Target_TotalScore"].apply(
            lambda x: "low" if x <= 10 else "medium" if x <= 15 else "high"
        )

        # Save results
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully generated {len(results)} transcripts")
        print(f"Results saved to: {OUTPUT_CSV_PATH}")

        # Print score distribution statistics
        print("\nScore Distribution Statistics (Total Scores):")
        print(df["Target_TotalScore"].describe())

        # Print quality level distribution with error handling
        print("\nQuality Level Distribution:")
        print(df["QualityLevel"].value_counts())
    else:
        print("Error: No transcripts were successfully generated")


if __name__ == "__main__":
    main()
