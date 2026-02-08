#!/usr/bin/env python3
"""
Simple direct test of checkpoint logic with minimal API calls.
Tests: 2 transcripts × 3 attempts = 6 total calls, checkpoint every 2
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Set environment
os.chdir(PROJECT_ROOT)
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set")
    sys.exit(1)

print("=" * 80)
print("CHECKPOINT LOGIC TEST - MINIMAL RUN")
print("=" * 80)
print("Config: 2 transcripts × 3 attempts = 6 API calls")
print("Checkpoint interval: 2 (should create 3 checkpoint files)")
print("=" * 80)
print()

# Modify Exp5 configuration temporarily
exp5_dir = PROJECT_ROOT / "experiments/exp5_final_validation/scripts"
exp5_script = exp5_dir / "01_run_final_validation.py"

# Read original
with open(exp5_script, 'r') as f:
    original_content = f.read()

# Create modified version
modified_content = original_content.replace(
    "NUM_TRANSCRIPTS_TO_USE = 250",
    "NUM_TRANSCRIPTS_TO_USE = 2  # TEST"
).replace(
    "NUM_ATTEMPTS_PER_TRANSCRIPT = 20",
    "NUM_ATTEMPTS_PER_TRANSCRIPT = 3  # TEST"
).replace(
    "checkpoint_interval = 100",
    "checkpoint_interval = 2  # TEST"
)

# Save modified version
test_script = PROJECT_ROOT / "test_exp5_run.py"
with open(test_script, 'w') as f:
    f.write(modified_content)

print(f"Running test experiment...")
print()

# Run test
import subprocess
result = subprocess.run(
    [sys.executable, str(test_script)],
    env=os.environ.copy()
)

print()
print("=" * 80)
print("VERIFYING RESULTS")
print("=" * 80)

# Check checkpoint files
results_dir = PROJECT_ROOT / "experiments/exp5_final_validation/results/processed_scores"
checkpoint_files = sorted(results_dir.glob("exp5_gpt5nano_*_batch*.csv"))

print(f"\nCheckpoint files found: {len(checkpoint_files)}")

if not checkpoint_files:
    print("❌ NO CHECKPOINT FILES CREATED")
    test_script.unlink()
    sys.exit(1)

import pandas as pd
total_rows = 0
for i, file in enumerate(checkpoint_files, 1):
    df = pd.read_csv(file)
    rows = len(df)
    total_rows += rows
    print(f"  {i}. {file.name}: {rows} rows")

expected = 6  # 2 transcripts × 3 attempts
print(f"\nTotal rows: {total_rows}")
print(f"Expected: {expected}")

# Cleanup
test_script.unlink()

if total_rows == expected:
    print("\n✅ TEST PASSED - Checkpoint logic works!")
    print("Ready to run full experiments.")
else:
    print(f"\n❌ TEST FAILED - Data loss: {total_rows}/{expected} rows")
    sys.exit(1)

