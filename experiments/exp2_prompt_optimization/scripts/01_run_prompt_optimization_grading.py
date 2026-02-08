#!/usr/bin/env python3
"""
Experiment 2: Prompt Strategy Optimization (using gpt-5-nano)

Compares Zero-Shot, Few-Shot, and Chain-of-Thought prompting strategies.

Configuration:
- Model: gpt-5-nano
- Transcripts: 50
- Attempts per strategy: 20
- Total: 3,000 grading attempts (50 × 20 × 3)
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
from experiments.common.models import DEFAULT_FINAL_MODEL, TEMPERATURES, get_max_workers
from dotenv import load_dotenv

load_dotenv()  # Loads from .env in project root

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configuration
EXPERIMENT_ID = "EXP2_PromptOptimization_GPT5Nano"
MODEL = DEFAULT_FINAL_MODEL
TEMPERATURE = TEMPERATURES["optimization"]
NUM_TRANSCRIPTS = 50
NUM_ATTEMPTS_PER_STRATEGY = 20
MAX_WORKERS = get_max_workers()

# Paths
DATA_PARTITIONS_FILE = PROJECT_ROOT / "experiments/dataset_partitions.json"
TRANSCRIPT_DIR = PROJECT_ROOT / "data/cleaned"
PROMPTS_DIR = PROJECT_ROOT / "experiments/exp2_prompt_optimization/prompts"
RESULTS_DIR = PROJECT_ROOT / "experiments/exp2_prompt_optimization/results/processed_scores"
RUBRIC_FILE = PROJECT_ROOT / "experiments/common/rubric_v5.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE = RESULTS_DIR / f"exp2_gpt5nano_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Prompt strategies
STRATEGIES = {
    "zero_shot": PROMPTS_DIR / "p2_1_zero_shot_prompt.txt",
    "few_shot": PROMPTS_DIR / "p2_2_few_shot_prompt.txt",
    "chain_of_thought": PROMPTS_DIR / "p2_3_cot_prompt.txt",
}


def create_llm_client():
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE, timeout=60.0, max_retries=3)


def run_single_grading(attempt_data):
    transcript_id = attempt_data["transcript_id"]
    strategy_name = attempt_data["strategy_name"]
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
            "prompt_strategy": strategy_name,
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
            "prompt_strategy": strategy_name,
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": None,
            "latency": None,
            "Parsing_Error": f"Error: {str(e)}",
        }


def load_existing_results():
    existing_results = []
    for file in RESULTS_DIR.glob("exp2_gpt5nano_checkpoint_*.csv"):
        try:
            df = pd.read_csv(file)
            existing_results.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
    
    if existing_results:
        combined = pd.concat(existing_results, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["transcript_id", "prompt_strategy", "attempt_num"], keep="last"
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
    print("EXPERIMENT 2: PROMPT OPTIMIZATION (GPT-5-NANO)")
    print("=" * 80)
    print(f"Model: {MODEL}")
    print(f"Strategies: {list(STRATEGIES.keys())}")
    print(f"Workers: {MAX_WORKERS}")
    print()
    
    # Load transcripts
    with open(DATA_PARTITIONS_FILE) as f:
        partitions = json.load(f)
    
    set_a = partitions["exp2_set_A"]
    selected_transcripts = set_a[:NUM_TRANSCRIPTS]
    print(f"Selected {len(selected_transcripts)} transcripts")
    
    # Load rubric
    rubric_text = load_text_file(str(RUBRIC_FILE))
    
    # Load existing results
    existing_df = load_existing_results()
    completed = set()
    if not existing_df.empty:
        completed = set(
            zip(existing_df["transcript_id"], existing_df["prompt_strategy"], existing_df["attempt_num"])
        )
        print(f"Resuming: {len(completed)} attempts completed")
    
    # Prepare all attempts
    all_attempts = []
    for strategy_name, prompt_file in STRATEGIES.items():
        prompt_template = load_text_file(str(prompt_file))
        for transcript_id in selected_transcripts:
            for attempt_num in range(1, NUM_ATTEMPTS_PER_STRATEGY + 1):
                if (transcript_id, strategy_name, attempt_num) not in completed:
                    all_attempts.append({
                        "transcript_id": transcript_id,
                        "strategy_name": strategy_name,
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
                    print(f"\nError: {attempt['transcript_id']} {attempt['strategy_name']} #{attempt['attempt_num']}: {e}")
                pbar.update(1)
    
    if results:
        batch_num += 1
        save_checkpoint(results, batch_num)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
