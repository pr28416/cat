#!/bin/bash
# Quick test of Exp5 with minimal data

cd /Users/pranavramesh/Developer/cat
source .venv/bin/activate

# Create a temporary modified script
cat > /tmp/test_exp5_mini.py << 'EOFPYTHON'
#!/usr/bin/env python3
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/pranavramesh/Developer/cat")
sys.path.append(str(PROJECT_ROOT))

# Import after path setup
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from experiments.common.file_utils import load_text_file
from experiments.common.parsing_utils import parse_scores_from_response
from experiments.common.enums import GradingCondition
from experiments.common.models import DEFAULT_FINAL_MODEL, TEMPERATURES, get_max_workers
from dotenv import load_dotenv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datetime import datetime

load_dotenv()

# TEST CONFIGURATION
NUM_TRANSCRIPTS_TO_USE = 2
NUM_ATTEMPTS_PER_TRANSCRIPT = 3
CHECKPOINT_INTERVAL = 2

MODEL = DEFAULT_FINAL_MODEL
TEMPERATURE = TEMPERATURES["final_validation"]
MAX_WORKERS = 4

DATA_PARTITIONS_FILE = PROJECT_ROOT / "experiments/dataset_partitions.json"
TRANSCRIPT_DIR = PROJECT_ROOT / "data/cleaned"
PROMPT_FILE = PROJECT_ROOT / "experiments/exp2_prompt_optimization/prompts/p2_2_few_shot_prompt.txt"
RESULTS_DIR = PROJECT_ROOT / "experiments/exp5_final_validation/results/processed_scores"
RUBRIC_FILE = PROJECT_ROOT / "experiments/common/rubric_v5.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_BASE = RESULTS_DIR / f"TEST_exp5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def create_llm_client():
    return ChatOpenAI(model=MODEL, temperature=TEMPERATURE, timeout=60.0, max_retries=3)

def run_single_grading_attempt(attempt_data):
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
            raise Exception(f"API error: {e}")
        
        parsed_scores = parse_scores_from_response(
            raw_response, GradingCondition.RUBRIC_BASED, transcript_id, attempt_num
        )
        
        return {
            "transcript_id": transcript_id,
            "model_name": MODEL,
            "attempt_num": attempt_num,
            "temperature": TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
            "raw_response": raw_response[:100],  # Truncate for test
            "latency": latency,
            **parsed_scores,
        }
    except Exception as e:
        return {
            "transcript_id": transcript_id,
            "model_name": MODEL,
            "attempt_num": attempt_num,
            "Parsing_Error": f"Error: {str(e)}",
        }

def save_checkpoint(results, batch_num):
    checkpoint_file = Path(str(CHECKPOINT_BASE) + f"_batch{batch_num:04d}.csv")
    df = pd.DataFrame(results)
    df.to_csv(checkpoint_file, index=False)
    print(f"✓ Checkpoint saved: {checkpoint_file.name} ({len(results)} rows)")

# Load data
with open(DATA_PARTITIONS_FILE) as f:
    partitions = json.load(f)

set_c = partitions["exp5_set_C"]
selected_transcripts = set_c[:NUM_TRANSCRIPTS_TO_USE]

prompt_template = load_text_file(str(PROMPT_FILE))
rubric_text = load_text_file(str(RUBRIC_FILE))

# Prepare attempts
all_attempts = []
for transcript_id in selected_transcripts:
    for attempt_num in range(1, NUM_ATTEMPTS_PER_TRANSCRIPT + 1):
        all_attempts.append({
            "transcript_id": transcript_id,
            "prompt_template": prompt_template,
            "rubric_text": rubric_text,
            "attempt_num": attempt_num,
        })

print(f"TEST: {len(all_attempts)} attempts, checkpoint every {CHECKPOINT_INTERVAL}")

# Run with checkpointing
results = []
batch_num = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(run_single_grading_attempt, attempt): attempt for attempt in all_attempts}
    
    with tqdm(total=len(all_attempts), desc="Grading") as pbar:
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                if (i + 1) % CHECKPOINT_INTERVAL == 0:
                    batch_num += 1
                    save_checkpoint(results, batch_num)
                    results = []
                    
            except Exception as e:
                print(f"Error: {e}")
            
            pbar.update(1)

if results:
    batch_num += 1
    save_checkpoint(results, batch_num)

print(f"\n✓ Test complete - created {batch_num} checkpoint files")
EOFPYTHON

python3 /tmp/test_exp5_mini.py

