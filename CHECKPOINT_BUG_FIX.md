# Checkpoint Bug Fix Required

## Problem

The checkpoint logic in all experiment runners has a critical bug that causes ~90% data loss:

```python
# CURRENT (BROKEN):
results = []
for i, future in enumerate(as_completed(futures)):
    result = future.result()
    results.append(result)

    if (i + 1) % 100 == 0:
        save_checkpoint(results)  # Saves only current batch
        results = []  # CLEARS BUFFER - loses all previous data!
```

## Impact

- **Exp2**: Only last ~600 rows saved (expected 3,000)
- **Exp3**: Only last ~700 rows saved (expected 2,800)
- **Exp5**: Only last ~600 rows saved (expected 5,000)

## Solution

Change checkpoint logic to:

1. **Append to file** instead of overwriting
2. OR use **unique timestamped filenames** for each checkpoint batch
3. OR **load existing + append** before saving

### Recommended Fix (Option 2 - Unique Filenames)

```python
checkpoint_counter = 0
results = []

for i, future in enumerate(as_completed(futures)):
    result = future.result()
    results.append(result)

    if (i + 1) % 100 == 0:
        checkpoint_counter += 1
        checkpoint_file = RESULTS_DIR / f"checkpoint_batch_{checkpoint_counter:04d}.csv"
        save_checkpoint(results, checkpoint_file)
        results = []  # Safe to clear now since we used unique filename

# Final save
if results:
    checkpoint_counter += 1
    checkpoint_file = RESULTS_DIR / f"checkpoint_batch_{checkpoint_counter:04d}.csv"
    save_checkpoint(results, checkpoint_file)
```

## Files to Fix

1. `/experiments/exp2_prompt_optimization/scripts/01_run_prompt_optimization_grading.py`
2. `/experiments/exp3_model_comparison/scripts/01_run_model_comparison_grading.py`
3. `/experiments/exp5_final_validation/scripts/01_run_final_validation.py`
4. `/experiments/exp7_gpt5_robustness/scripts/01_run_gpt5_grading.py`

## Verification Steps

After fix:

1. Run small test (10 transcripts × 2 attempts)
2. Verify multiple checkpoint files created
3. Verify total rows = expected count
4. Then run full experiments

## Analysis Scripts Updated

✅ Exp5 analysis script already updated to load and combine multiple checkpoint files
✅ Exp2/Exp3 analysis scripts work with current data (need update for multiple files)
