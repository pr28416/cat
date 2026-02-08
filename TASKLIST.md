# CHAT Research Project - Task List

## Phase 0: Task Tracking Setup

- [x] Create TASKLIST.md with checkboxed tasks

## Phase 1: Repository Cleanup

- [x] Delete planning/status markdown files (REFACTORING_SUMMARY.md, CLEANUP_PLAN.md, NEXT_STEPS.md, HONEST_STATUS.md)
- [x] Remove unused scripts and files
- [x] Review supplements directory and keep only what's referenced in paper
- [x] Clean up experiment directories

## Phase 2: LangChain Dependencies

- [x] Update requirements.txt to add langchain and langchain-openai
- [x] Ensure .venv exists and install dependencies

## Phase 3: Model Constants & Environment Setup

- [x] Create experiments/common/models.py with model constants
- [x] Verify OPENAI_API_KEY environment variable setup
- [x] Remove multi-key and base URL logic from scripts

## Phase 4: LangChain Migration

- [x] Update Exp2 runner to use LangChain directly
- [x] Update Exp3 runner to use LangChain directly
- [x] Update Exp5 runner to use LangChain directly
- [x] Create Exp7 runner using LangChain
- [x] Remove or deprecate ai/openai.py

## Phase 5: Multithreading Support

- [x] Add ThreadPoolExecutor to Exp2 runner (6-8 workers)
- [x] Add ThreadPoolExecutor to Exp3 runner (6-8 workers)
- [x] Add ThreadPoolExecutor to Exp5 runner (6-8 workers)
- [x] Add ThreadPoolExecutor to Exp7 runner (6-8 workers)
- [x] Implement checkpointing logic for resumable runs
- [x] Add retry/backoff logic for rate limits

## Phase 6: Run Experiments

### Exp2: Prompt Optimization (gpt-5-nano)

- [~] Run Exp2: 50 transcripts × 20 attempts × 3 strategies = 3,000 calls
- [ ] Analyze Exp2 results
- [ ] Generate Exp2 results markdown and plots

### Exp3: Model Comparison (include GPT-5 models)

- [ ] Run Exp3: 40 transcripts × 10 attempts × 6 models = 2,400 calls
- [ ] Analyze Exp3 results
- [ ] Generate Exp3 results markdown and plots

### Exp5: Final Reliability (gpt-5-nano)

- [ ] Run Exp5: 250 transcripts × 20 attempts = 5,000 calls
- [ ] Analyze Exp5 reliability results
- [ ] Run Exp5 known-groups validity analysis
- [ ] Generate Exp5 results markdowns and plots

### Exp7: GPT-5 Family Robustness

- [ ] Run Exp7: 50 transcripts × 20 attempts × gpt-5-mini = 1,000 calls
- [ ] Analyze Exp7 results
- [ ] Generate Exp7 results markdown and plots

## Phase 6.5: Fix Checkpoint Bug & Rerun

- [x] Fix checkpoint logic in all 4 experiment scripts (use unique filenames per batch)
- [x] Update analysis scripts to combine multiple checkpoint files
- [x] Test with small run (6 API calls) - ✅ VERIFIED WORKING
- [ ] Rerun Exp2 (3,000 calls, ~45 min)
- [ ] Rerun Exp3 (2,800 calls, ~45 min)
- [ ] Rerun Exp5 (5,000 calls, ~75 min)
- [ ] Run Exp7 (1,000 calls, ~15 min)
- [ ] Run all analysis scripts and generate plots
- [ ] Run Exp5 known-groups validity analysis

## Phase 7: Final Documentation

- [ ] Update paper with gpt-5-nano as default
- [ ] Update all paper results/tables from reruns
- [ ] Rewrite README.md with setup and run instructions
- [x] Create CHANGELOG.md documenting all changes
- [ ] Final review: ensure no extra markdown files remain

---

**Status Legend:**

- [ ] Not started
- [x] Completed
- [~] In progress (marked with ~)

**Last Updated:** Starting execution
