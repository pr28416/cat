#!/usr/bin/env python3
"""
Experiment 5: Final Tool Validation (using gpt-5-nano)

This script runs the final validation of CHAT using gpt-5-nano as the default model.

Configuration:
- Model: gpt-5-nano (cost-optimal GPT-5 model)
- Prompt: Few-Shot strategy
- Dataset: 250 transcripts (125 DoVA + 125 Nature)
- Attempts: 20 per transcript
- Total: 5,000 grading attempts
- Concurrency: 6-8 workers (configurable via CHAT_MAX_WORKERS)
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition
from experiments.common.models import DEFAULT_FINAL_MODEL, TEMPERATURES, get_max_workers
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Loads from .env in project root

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# --- Experiment Configuration ---
EXPERIMENT_ID = "EXP5_FinalToolValidation_GPT5Nano"
MODEL = DEFAULT_FINAL_MODEL  # "gpt-5-nano"
TEMPERATURE = TEMPERATURES["final_validation"]  # 0.3
NUM_TRANSCRIPTS_TO_USE = 250
NUM_ATTEMPTS_PER_TRANSCRIPT = 20
MAX_WORKERS = get_max_workers()  # 6-8

# --- File Paths ---
DATA_PARTITIONS_FILE = PROJECT_ROOT / "experiments/dataset_partitions.json"
TRANSCRIPT_DIR = PROJECT_ROOT / "data/cleaned"
PROMPT_FILE = PROJECT_ROOT / "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt"
RESULTS_DIR = PROJECT_ROOT / "experiments/exp5_final_validation/results/processed_scores"
RUBRIC_FILE = PROJECT_ROOT / "experiments/common/rubric_v5.md"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint file base (will append batch number)
CHECKPOINT_BASE = RESULTS_DIR / f"exp5_gpt5nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_llm_client():
    """Create a LangChain ChatOpenAI client with retry settings."""
    return ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
        timeout=60.0,
        max_retries=3,
    )


def run_single_grading_attempt(attempt_data):
    """
    Run a single grading attempt using LangChain.

    Args:
        attempt_data (dict): Contains transcript_id, prompt_template, rubric_text, attempt_num, llm

    Returns:
        dict: Results including scores, timing, and any errors
    """
    transcript_id = attempt_data["transcript_id"]
    prompt_template = attempt_data["prompt_template"]
    rubric_text = attempt_data["rubric_text"]
    attempt_num = attempt_data["attempt_num"]

    try:
        # Load transcript
        transcript_text = load_text_file(str(TRANSCRIPT_DIR / transcript_id))

        # Format the prompt
        prompt = prompt_template.replace("{{rubric}}", rubric_text).replace(
            "{{transcript}}", transcript_text
        )
        
        # Create LLM client for this attempt (thread-safe)
        llm = create_llm_client()

        # Get LLM completion with timing
        start_time = time.time()
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_response = response.content
            latency = time.time() - start_time
        except Exception as e:
            latency = time.time() - start_time
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                # Add exponential backoff for rate limits
                time.sleep(min(60, 2 ** attempt_num))
                raise Exception(f"Rate limit error: {error_msg}")
            raise Exception(f"API error: {error_msg}")
        
        # Parse the response
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )

        return {
            "experiment_id": EXPERIMENT_ID,
            "transcript_id": transcript_id,
            "model_name": MODEL,
            "prompt_strategy": "few_shot",
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response,
            "latency": latency,
            **parsed_scores,
        }

    except Exception as e:
        return {
            "experiment_id": EXPERIMENT_ID,
            "transcript_id": transcript_id,
            "model_name": MODEL,
            "prompt_strategy": "few_shot",
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": None,
            "latency": None,
            "Parsing_Error": f"Error: {str(e)}",
        }


def load_existing_results():
    """Load existing checkpoint files to resume from where we left off."""
    existing_results = []
    for file in RESULTS_DIR.glob("exp5_gpt5nano_checkpoint_*.csv"):
        try:
            df = pd.read_csv(file)
            existing_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if existing_results:
        combined = pd.concat(existing_results, ignore_index=True)
        # Remove duplicates (keep last)
        combined = combined.drop_duplicates(
            subset=["transcript_id", "attempt_num"], keep="last"
        )
        return combined
    return pd.DataFrame()


def save_checkpoint(results, batch_num):
    """Save results to unique checkpoint file."""
    checkpoint_file = Path(str(CHECKPOINT_BASE) + f"_batch{batch_num:04d}.csv")
    df = pd.DataFrame(results)
    df.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved: {checkpoint_file} ({len(results)} rows)")


def main():
    print("=" * 80)
    print("EXPERIMENT 5: FINAL VALIDATION (GPT-5-NANO)")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Transcripts: {NUM_TRANSCRIPTS_TO_USE}")
    print(f"Attempts per transcript: {NUM_ATTEMPTS_PER_TRANSCRIPT}")
    print(f"Workers: {MAX_WORKERS}")
    print(f"Total grading attempts: {NUM_TRANSCRIPTS_TO_USE * NUM_ATTEMPTS_PER_TRANSCRIPT}")
    print()
    
    # Load partition data
    with open(DATA_PARTITIONS_FILE) as f:
        partitions = json.load(f)

    # Get Set C transcripts (125 DoVA + 125 Nature)
    set_c = partitions["exp5_set_C"]
    dova_transcripts = [t for t in set_c if t.startswith("DOVA_")]
    nature_transcripts = [t for t in set_c if t.startswith("NATURE_")]
    
    # Balance the sets
    selected_transcripts = dova_transcripts[:125] + nature_transcripts[:125]
    print(f"Selected {len(selected_transcripts)} transcripts (125 DoVA + 125 Nature)")

    # Load prompt template and rubric
    prompt_template = load_text_file(str(PROMPT_FILE))
    rubric_text = load_text_file(str(RUBRIC_FILE))
    
    # Load existing results for checkpointing
    existing_df = load_existing_results()
    completed = set()
    if not existing_df.empty:
        completed = set(
            zip(existing_df["transcript_id"], existing_df["attempt_num"])
        )
        print(f"Resuming: {len(completed)} attempts already completed")
    
    # Prepare all attempts
    all_attempts = []
    for transcript_id in selected_transcripts:
        for attempt_num in range(1, NUM_ATTEMPTS_PER_TRANSCRIPT + 1):
            if (transcript_id, attempt_num) not in completed:
                all_attempts.append({
                    "transcript_id": transcript_id,
                    "prompt_template": prompt_template,
                    "rubric_text": rubric_text,
                    "attempt_num": attempt_num,
                })
    
    print(f"Running {len(all_attempts)} new grading attempts...")
    print()
    
    # Run grading attempts with multithreading
    results = []
    checkpoint_interval = 100  # Save every 100 completions
    batch_num = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(run_single_grading_attempt, attempt): attempt
            for attempt in all_attempts
        }
        
        with tqdm(total=len(all_attempts), desc="Grading") as pbar:
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Periodic checkpointing with unique filenames
                    if (i + 1) % checkpoint_interval == 0:
                        batch_num += 1
                        save_checkpoint(results, batch_num)
                        results = []  # Clear to save memory (safe now with unique files)
                        
                except Exception as e:
                    attempt = futures[future]
                    print(f"\nError in attempt: {attempt['transcript_id']} #{attempt['attempt_num']}: {e}")
                
                pbar.update(1)
    
    # Final save
    if results:
        batch_num += 1
        save_checkpoint(results, batch_num)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nNext step: Run 02_analyze_final_validation.py to generate analysis")


if __name__ == "__main__":
    main()
