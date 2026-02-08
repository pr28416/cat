# Changelog

## [2.0.0] - 2026-01-04

### Major Changes

#### Model Migration

- **Switched default model from GPT-4o to gpt-5-nano**
  - gpt-5-nano is now the primary model for all CHAT assessments
  - Legacy models (GPT-4o, GPT-4o-mini, GPT-4.1, o3, o3-mini) retained only for comparison experiments
  - All main results and paper claims now based on gpt-5-nano performance

#### LangChain Integration

- **Migrated all LLM calls from custom wrapper to LangChain**
  - Replaced `ai/openai.py` with direct `langchain-openai` usage
  - Consistent error handling and retry logic across all experiments
  - Better abstractions for model switching and configuration

#### Multithreading Support

- **Added concurrent execution with 6-8 workers (configurable)**
  - ThreadPoolExecutor-based parallel processing for all experiments
  - Configurable via `CHAT_MAX_WORKERS` environment variable
  - Significantly reduced runtime for large-scale experiments

#### Checkpointing & Resumability

- **All experiments now support checkpointing**
  - Automatic checkpoint saves every 100 completions
  - Can resume from checkpoint after interruption
  - Skips already-completed (transcript, model, attempt) tuples

### Experiments Rerun

#### Experiment 2: Prompt Optimization (gpt-5-nano)

- Rerun with gpt-5-nano as default model
- 50 transcripts × 20 attempts × 3 strategies = 3,000 calls
- Results: `experiments/exp2_prompt_optimization/results/`

#### Experiment 3: Model Comparison (including GPT-5)

- Added gpt-5-nano and gpt-5-mini to model comparison
- Compared 6 models total (2 GPT-5 + 4 legacy)
- 40 transcripts × 10 attempts × 6 models = 2,400 calls
- Results: `experiments/exp3_model_comparison/results/`

#### Experiment 5: Final Validation (gpt-5-nano)

- **Full rerun**: 250 transcripts × 20 attempts = 5,000 calls
- Primary evidence for CHAT reliability with gpt-5-nano
- Includes known-groups validity analysis (DoVA vs Nature)
- Results: `experiments/exp5_final_validation/results/`

#### Experiment 7: GPT-5 Family Robustness

- New experiment comparing gpt-5-mini vs gpt-5-nano baseline
- 50 transcripts × 20 attempts = 1,000 calls
- Results: `experiments/exp7_gpt5_robustness/results/`

### Repository Cleanup

- **Removed non-deliverable files**
  - Deleted planning documents (REFACTORING_SUMMARY.md, CLEANUP_PLAN.md, etc.)
  - Removed test utilities and preprocessing scripts
  - Cleaned up old experiment results (will be regenerated)
- **Streamlined structure**
  - Only essential files remain: README, CHANGELOG, TASKLIST, paper, and experiment results
  - Supplements directory for supplementary materials
  - Clear separation between main experiments and supporting analyses

### Configuration Changes

- **Single API key**: Now uses only `OPENAI_API_KEY` (no multi-key support)
- **No custom base URL**: Uses default OpenAI endpoint
- **Environment**: Loads from `.env` file in project root
- **Model constants**: Centralized in `experiments/common/models.py`

### Dependencies

- Added `langchain>=0.1.0`
- Added `langchain-openai>=0.0.5`
- Removed dependency on custom `ai/openai.py` module

### Paper Updates

- Updated all claims to reflect gpt-5-nano as default model
- Revised results tables and figures based on new experiment runs
- Added GPT-5 family robustness analysis
- Maintained legacy model comparisons for completeness

### Breaking Changes

- **API**: Custom `ai/openai.py` module deprecated
- **Models**: GPT-4o is no longer the default (gpt-5-nano is)
- **Configuration**: Multi-key setup no longer supported
- **Experiments**: All previous results replaced with gpt-5-nano runs

---

## [1.0.0] - 2025-07-08

### Initial Release

- CHAT tool for oral health literacy assessment
- GPT-4o as primary model
- Experiments 1-5 completed
- Paper draft completed
- Repository structure established
