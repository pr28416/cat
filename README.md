# LLM-based Patient Health Communication Assessment

This project evaluates the utility of Large Language Models (LLMs) for assessing patient health communication using a standardized rubric.

## Project Structure

```
.
├── ai/                      # AI integration utilities
│   └── openai.py            # OpenAI API integration
├── data/                    # Data storage
│   ├── cleaned/             # Cleaned real transcripts (for Exp 2, 3, 5)
│   └── synthetic/           # Generated synthetic transcripts (for Exp 1)
├── experiments/             # Experiment code and resources
│   ├── common/              # Shared utilities (rubric, enums, utils)
│   ├── exp1_baseline_utility/  # Experiment 1: Baseline Utility
│   │   ├── prompts/
│   │   ├── results/
│   │   └── scripts/
│   ├── exp2_prompt_optimization/ # Experiment 2: Prompt Optimization
│   │   └── ... (similar structure)
│   ├── exp3_model_comparison/   # Experiment 3: LLM Architecture Comparison
│   │   └── ...
│   ├── exp4_reasoning_analysis/ # Experiment 4: Qualitative Reasoning Analysis
│   │   └── ...
│   ├── exp5_optimized_tool_validation/ # Experiment 5: Optimized Tool Validation
│   │   └── ...
│   └── README.md              # README for each experiment (planned)
├── prd.md                   # Product Requirements Document
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup

1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Create a `.env.local` file with your OpenAI API credentials:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com
```

## Experiment Status

| Experiment   | Description                                           | Status                                                                                                                                               |
| ------------ | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Experiment 1 | Baseline LLM-Rubric Utility & Synthetic Data Efficacy | Done (Results: [experiments/exp1_baseline_utility/results/experiment1_results.md](experiments/exp1_baseline_utility/results/experiment1_results.md)) |
| Experiment 2 | Prompt Strategy Optimization                          | Planned                                                                                                                                              |
| Experiment 3 | LLM Architecture Comparison                           | Planned                                                                                                                                              |
| Experiment 4 | Qualitative Reasoning Analysis                        | Planned                                                                                                                                              |
| Experiment 5 | Optimized Tool Validation                             | Planned                                                                                                                                              |

## Running Experiment 1: Baseline Utility Assessment

This experiment evaluates the baseline utility of LLMs for synthetic transcript generation and assessment.

### Quick Start

Run the experiment using the provided script:

```bash
./run_experiment1.sh
```

### Manual Steps

If you prefer to run steps individually:

1. Generate synthetic transcripts with target scores:

```bash
cd experiments/exp1_baseline_utility/scripts
python 01_generate_synthetic_transcripts.py
```

2. Grade the synthetic transcripts using both non-rubric and rubric-based approaches:

```bash
python 02_run_grading.py
```

3. Analyze the results to compare grading approaches:

```bash
python 03_analyze_results.py
```

## Patient Health Communication Rubric

The project uses the "Patient Health Communication Rubric v5.0" with 5 categories:

1. Clarity of Language
2. Lexical Diversity
3. Conciseness and Completeness
4. Engagement with Health Information
5. Health Literacy Indicator

Each category is scored from 1 (Poor) to 4 (Excellent), resulting in a total score range of 5-20.

## Future Experiments

Additional experiments will be implemented to evaluate:

- Experiment 2: Prompt Strategy Optimization (Zero-shot, Few-shot, CoT)
- Experiment 3: LLM Architecture Comparison (performance of different LLMs)
- Experiment 4: Qualitative Reasoning Analysis (alignment with rubric, evidence citation)
- Experiment 5: Optimized Tool Validation (reliability on unseen real transcripts, uncertainty quantification)
