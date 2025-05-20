# Repository File Structure Plan

This document outlines the planned file structure for the LLM-driven patient health communication assessment project.

## Root Directory

- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `README.md`: General overview of the project, setup instructions, etc.
- `prd.md`: The Project Requirements Document.
- `file_structure.md`: This file.
- `requirements.txt`: Python package dependencies.

## `data/` Directory

For all datasets used in the project.

- `data/raw/`: Original, unprocessed transcript data.
  - `data/raw/dova/`: Raw DoVA transcripts.
  - `data/raw/nature_paper/`: Raw transcripts from the Nature paper.
- `data/cleaned/`: Standardized plain text UTF-8 transcripts (e.g., `DOVA_H986.txt`).
- `data/synthetic/`: Synthetic transcripts generated in Experiment 1.
  - `data/synthetic/exp1_synthetic_transcripts.csv` (or individual `.txt` files with metadata).
- `data/sets/`: Lists of `TranscriptIDs` for data partitions (Set A, Set B, Set C).
  - `data/sets/exp2_set_A_transcript_ids.txt`
  - `data/sets/exp3_set_B_transcript_ids.txt`
  - `data/sets/exp5_set_C_transcript_ids.txt`

## `experiments/` Directory

Houses all code, prompts, and results specific to each experiment.

- `experiments/common/`: Shared utilities (API interaction, score parsing, rubric embedding).
  - `experiments/common/llm_utils.py`
  - `experiments/common/rubric_utils.py`
  - `experiments/common/analysis_utils.py`
- `experiments/exp1_baseline_utility/`
  - `scripts/`: Python scripts for generating synthetic data and running grading.
  - `prompts/`: Text files for P1.1 (Non-Rubric) and P1.2 (Rubric-Based) prompts.
  - `results/`:
    - `raw_api_logs/`: Logs of all API calls.
    - `processed_scores/`: Parsed scores from LLM outputs.
    - `analysis/`: STDEVs, MAEs, statistical test results, plots.
- `experiments/exp2_prompt_optimization/`
  - `scripts/`: Python scripts for running grading attempts.
  - `prompts/`: Text files for P2.1 (Zero-shot), P2.2 (Few-shot), P2.3 (CoT) prompts. Include few-shot examples here.
  - `results/`: Sub-structured like Exp1 for logs, scores, and analysis.
- `experiments/exp3_model_comparison/`
  - `scripts/`: Python scripts.
  - `prompts/`: The winning prompt strategy from Exp2.
  - `results/`: Sub-structured as above.
- `experiments/exp4_reasoning_analysis/`
  - `scripts/`: Scripts for applying the coding scheme (if automated).
  - `materials/`: The finalized Reasoning Coding Scheme.
  - `results/`: Coded reasoning segments, frequency analyses.
- `experiments/exp5_final_validation/`
  - `scripts/`: Python scripts.
  - `prompts/`: The optimized tool configuration (winning LLM and prompt).
  - `results/`: Sub-structured as above, containing benchmark performance.

## `notebooks/` Directory

For Jupyter notebooks used for exploratory data analysis (EDA).

- `notebooks/exp1_eda.ipynb`
- `notebooks/exp2_eda.ipynb`
- (etc. for other experiments)

## `results_summary/` Directory

For aggregated results, key figures, and tables for the final paper.

- `results_summary/figures/`
- `results_summary/tables/`

## `paper/` Directory

For the manuscript and supplementary materials.

- `paper/manuscript.md` (or `.docx`)
- `paper/supplementary_materials/`
  - `paper/supplementary_materials/rubric_v5.md` (Full text of Rubric 5.0)
  - (Other materials like detailed prompt texts)
