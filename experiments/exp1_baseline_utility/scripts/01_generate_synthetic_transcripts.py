import os
import random
import sys
import pandas as pd
import time
import multiprocessing as mp
from functools import partial
import json

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
        json.dump(checkpoint_data, f)


def generate_target_scores(num_transcripts):
    """Generate target scores following the defined distribution."""
    scores = []
    num_low = int(num_transcripts * 0.2)  # 20% low scores
    num_medium = int(num_transcripts * 0.4)  # 40% medium scores
    num_high = num_transcripts - num_low - num_medium  # Remaining high scores

    # Generate scores for each category
    scores.extend([random.randint(*SCORE_DISTRIBUTION["low"]) for _ in range(num_low)])
    scores.extend(
        [random.randint(*SCORE_DISTRIBUTION["medium"]) for _ in range(num_medium)]
    )
    scores.extend(
        [random.randint(*SCORE_DISTRIBUTION["high"]) for _ in range(num_high)]
    )

    # Shuffle the scores
    random.shuffle(scores)
    return scores


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
    """Generate a single synthetic transcript."""
    i, generation_prompt_template, rubric_text, target_score = args
    synthetic_transcript_id = f"SYNTH_EXP1_{i+1:03d}"

    print(
        f"\nGenerating transcript {i+1}/{NUM_SYNTHETIC_TRANSCRIPTS} (ID: {synthetic_transcript_id}, Target Score: {target_score})"
    )

    # Construct the prompt with explicit quality level
    quality_level = (
        "low" if target_score <= 10 else "medium" if target_score <= 15 else "high"
    )
    prompt = generation_prompt_template.replace("{{TTS}}", str(target_score))
    prompt = prompt.replace("{{RubricV5Text}}", rubric_text)
    prompt = prompt.replace("{{QualityLevel}}", quality_level)

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
            result = {
                "SyntheticTranscriptID": synthetic_transcript_id,
                "TargetTotalScore": target_score,
                "QualityLevel": quality_level,
                "GeneratedTranscript": final_transcript_text,
                "FullLLMResponse": llm_response,
            }
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

    # Generate target scores for remaining transcripts
    remaining_count = NUM_SYNTHETIC_TRANSCRIPTS - len(completed_indices)
    target_scores = generate_target_scores(remaining_count)

    # Prepare arguments for parallel processing
    args = [
        (i, generation_prompt_template, rubric_text, target_scores[i])
        for i in range(remaining_count)
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
        completed_indices.extend(
            [
                i
                for i, r in zip(range(NUM_SYNTHETIC_TRANSCRIPTS), new_results)
                if r is not None
            ]
        )
        save_checkpoint(completed_indices, results)

    if results:
        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Ensure all required columns are present
        required_columns = [
            "SyntheticTranscriptID",
            "TargetTotalScore",
            "QualityLevel",
            "GeneratedTranscript",
            "FullLLMResponse",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Warning: Missing columns in results: {missing_columns}")
            # Add missing columns with default values
            for col in missing_columns:
                if col == "QualityLevel":
                    df[col] = df["TargetTotalScore"].apply(
                        lambda x: "low" if x <= 10 else "medium" if x <= 15 else "high"
                    )
                else:
                    df[col] = None

        # Save results
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSuccessfully generated {len(results)} transcripts")
        print(f"Results saved to: {OUTPUT_CSV_PATH}")

        # Print score distribution statistics
        print("\nScore Distribution Statistics:")
        print(df["TargetTotalScore"].describe())

        # Print quality level distribution with error handling
        try:
            print("\nQuality Level Distribution:")
            print(df["QualityLevel"].value_counts())
        except KeyError:
            print("\nWarning: Quality level distribution not available")
            # Calculate and print quality levels based on target scores
            quality_levels = df["TargetTotalScore"].apply(
                lambda x: "low" if x <= 10 else "medium" if x <= 15 else "high"
            )
            print("\nDerived Quality Level Distribution:")
            print(quality_levels.value_counts())
    else:
        print("Error: No transcripts were successfully generated")


if __name__ == "__main__":
    main()
