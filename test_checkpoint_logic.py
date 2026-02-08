#!/usr/bin/env python3
"""
Test script to verify checkpoint logic works correctly.

This runs a TINY experiment (2 transcripts × 3 attempts = 6 total calls)
with checkpoint_interval=2 to verify we get multiple checkpoint files.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Temporarily modify Exp5 script to use tiny parameters
import json

# Create test configuration
TEST_CONFIG = {
    "NUM_TRANSCRIPTS": 2,
    "NUM_ATTEMPTS": 3,
    "CHECKPOINT_INTERVAL": 2,  # Checkpoint every 2 completions
}

print("=" * 80)
print("CHECKPOINT LOGIC TEST")
print("=" * 80)
print(f"Configuration:")
print(f"  Transcripts: {TEST_CONFIG['NUM_TRANSCRIPTS']}")
print(f"  Attempts per transcript: {TEST_CONFIG['NUM_ATTEMPTS']}")
print(f"  Total API calls: {TEST_CONFIG['NUM_TRANSCRIPTS'] * TEST_CONFIG['NUM_ATTEMPTS']}")
print(f"  Checkpoint interval: {TEST_CONFIG['CHECKPOINT_INTERVAL']}")
print(f"  Expected checkpoint files: {(TEST_CONFIG['NUM_TRANSCRIPTS'] * TEST_CONFIG['NUM_ATTEMPTS']) // TEST_CONFIG['CHECKPOINT_INTERVAL'] + 1}")
print()

# Read the Exp5 script
exp5_script = PROJECT_ROOT / "experiments/exp5_final_validation/scripts/01_run_final_validation.py"
with open(exp5_script, 'r') as f:
    content = f.read()

# Create modified version for testing
test_content = content.replace(
    "NUM_TRANSCRIPTS_TO_USE = 250",
    f"NUM_TRANSCRIPTS_TO_USE = {TEST_CONFIG['NUM_TRANSCRIPTS']}"
).replace(
    "NUM_ATTEMPTS_PER_TRANSCRIPT = 20",
    f"NUM_ATTEMPTS_PER_TRANSCRIPT = {TEST_CONFIG['NUM_ATTEMPTS']}"
).replace(
    "checkpoint_interval = 100",
    f"checkpoint_interval = {TEST_CONFIG['CHECKPOINT_INTERVAL']}"
)

# Write temporary test script
test_script = PROJECT_ROOT / "test_exp5_tiny.py"
with open(test_script, 'w') as f:
    f.write(test_content)

print(f"Created test script: {test_script}")
print()
print("Running test experiment...")
print("=" * 80)
print()

# Run the test
import subprocess
result = subprocess.run(
    [sys.executable, str(test_script)],
    cwd=str(PROJECT_ROOT),
    capture_output=False
)

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)

# Check results
results_dir = PROJECT_ROOT / "experiments/exp5_final_validation/results/processed_scores"
checkpoint_files = sorted(results_dir.glob("exp5_gpt5nano_*_batch*.csv"))

print(f"\nCheckpoint files created: {len(checkpoint_files)}")
for i, file in enumerate(checkpoint_files, 1):
    import pandas as pd
    df = pd.read_csv(file)
    print(f"  {i}. {file.name}: {len(df)} rows")

# Verify total rows
if checkpoint_files:
    total_rows = sum(len(pd.read_csv(f)) for f in checkpoint_files)
    expected_rows = TEST_CONFIG['NUM_TRANSCRIPTS'] * TEST_CONFIG['NUM_ATTEMPTS']
    
    print(f"\nTotal rows across all checkpoints: {total_rows}")
    print(f"Expected rows: {expected_rows}")
    
    if total_rows == expected_rows:
        print("\n✅ TEST PASSED: All data preserved!")
    else:
        print(f"\n❌ TEST FAILED: Data loss detected ({total_rows}/{expected_rows})")
        sys.exit(1)
else:
    print("\n❌ TEST FAILED: No checkpoint files created")
    sys.exit(1)

# Cleanup
test_script.unlink()
print(f"\nCleaned up test script: {test_script}")
print("\n✅ Checkpoint logic verified - ready for full runs!")

