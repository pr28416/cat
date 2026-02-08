#!/usr/bin/env python3
"""
Experiment 7: GPT-5 Family Robustness

Compares gpt-5-mini against gpt-5-nano baseline (Exp5) for reliability consistency.

Configuration:
- Model: gpt-5-mini (comparison against Exp5's gpt-5-nano)
- Transcripts: 50 (stratified sample from Exp5)
- Attempts: 20 per transcript
- Total: 1,000 grading attempts
- Concurrency: 6-8 workers
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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition
from experiments.common.models import TEMPERATURES, get_max_workers
from dotenv import load_dotenv

load_dotenv()  # Loads from .env in project root

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configuration
EXPERIMENT_ID = "EXP7_GPT5_Robustness"
MODEL = "gpt-5-mini"  # Comparing against gpt-5-nano baseline
TEMPERATURE = TEMPERATURES["robustness"]
NUM_TRANSCRIPTS = 50
NUM_ATTEMPTS_PER_TRANSCRIPT = 20
MAX_WORKERS = get_max_workers()

# Paths
TRANSCRIPT_DIR = PROJECT_ROOT / "data/cleaned"
PROMPT_FILE = PROJECT_ROOT / "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt"
RESULTS_DIR = PROJECT_ROOT / "experiments/exp7_gpt5_robustness/results/processed_scores"
RUBRIC_FILE = PROJECT_ROOT / "experiments/common/rubric_v5.md"
SAMPLE_FILE = PROJECT_ROOT / "experiments/exp7_gpt5_robustness/selected_transcripts_for_gpt5_robustness.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE = RESULTS_DIR / f"exp7_gpt5mini_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def create_llm_client():
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE, timeout=60.0, max_retries=3)


def run_single_grading(attempt_data):
    transcript_id = attempt_data["transcript_id"]
    prompt_template = attempt_data["prompt_template"]
    rubric_text = attempt_data["rubric_text"]
    attempt_num = attempt_data["attempt_num"]
    
    try:
        transcript_text = load_text_file(str(TRANSCRIPT_DIR / transcript_id))
        prompt = prompt_template.replace("{{rubric}}", rubric_text).replace("{{transcript}}", transcript_text)
        
        llm = create_llm_client()
        start_time = time.time()
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            raw_response = response.content
            latency = time.time() - start_time
        except Exception as e:
            latency = time.time() - start_time
            if "rate_limit" in str(e).lower():
                time.sleep(min(60, 2 ** attempt_num))
                raise Exception(f"Rate limit: {e}")
            raise Exception(f"API error: {e}")
        
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
    existing_results = []
    for file in RESULTS_DIR.glob("exp7_gpt5mini_checkpoint_*.csv"):
        try:
            df = pd.read_csv(file)
            existing_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if existing_results:
        combined = pd.concat(existing_results, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["transcript_id", "attempt_num"], keep="last"
        )
        return combined
    return pd.DataFrame()


def save_checkpoint(results, batch_num):
    checkpoint_file = Path(str(CHECKPOINT_BASE) + f"_batch{batch_num:04d}.csv")
    df = pd.DataFrame(results)
    df.to_csv(checkpoint_file, index=False)
    print(f"Checkpoint saved: {checkpoint_file} ({len(results)} rows)")


def main():
    print("=" * 80)
    print("EXPERIMENT 7: GPT-5 FAMILY ROBUSTNESS")
    print("=" * 80)
    print(f"Model: {MODEL} (comparing against gpt-5-nano baseline)")
    print(f"Workers: {MAX_WORKERS}")
    print()
    
    # Check if sample file exists, if not run the selection script
    if not SAMPLE_FILE.exists():
        print("Sample file not found. Running selection script...")
        import subprocess
        subprocess.run([
            sys.executable,
            str(PROJECT_ROOT / "experiments/exp7_gpt5_robustness/00_select_robustness_sample.py")
        ], check=True)
    
    # Load selected transcripts
    sample_df = pd.read_csv(SAMPLE_FILE)
    selected_transcripts = sample_df["transcript_id"].tolist()[:NUM_TRANSCRIPTS]
    print(f"Selected {len(selected_transcripts)} transcripts")
    
    # Load prompt and rubric
    prompt_template = load_text_file(str(PROMPT_FILE))
    rubric_text = load_text_file(str(RUBRIC_FILE))
    
    # Load existing results
    existing_df = load_existing_results()
    completed = set()
    if not existing_df.empty:
        completed = set(
            zip(existing_df["transcript_id"], existing_df["attempt_num"])
        )
        print(f"Resuming: {len(completed)} attempts completed")
    
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
    
    print(f"Running {len(all_attempts)} grading attempts...")
    print()
    
    results = []
    batch_num = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_grading, attempt): attempt for attempt in all_attempts}
        
        with tqdm(total=len(all_attempts), desc="Grading") as pbar:
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 100 == 0:
                        batch_num += 1
                        save_checkpoint(results, batch_num)
                        results = []
                except Exception as e:
                    attempt = futures[future]
                    print(f"\nError: {attempt['transcript_id']} #{attempt['attempt_num']}: {e}")
                pbar.update(1)
    
    if results:
        batch_num += 1
        save_checkpoint(results, batch_num)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 7 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

