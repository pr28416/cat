# LLM-based Patient Health Communication Assessment

This project evaluates the utility of Large Language Models (LLMs) for assessing patient health communication using a standardized rubric.

## Project Structure

```
.
├── ai/                      # AI integration utilities
│   └── openai.py            # OpenAI API integration
├── data/                    # Data storage
│   ├── cleaned/             # Cleaned real transcripts
│   └── synthetic/           # Generated synthetic transcripts
├── experiments/             # Experiment code and resources
│   ├── common/              # Shared utilities and resources
│   └── exp1_baseline_utility/  # Experiment 1: Baseline Utility
│       ├── prompts/         # Prompts for generation and evaluation
│       ├── results/         # Results storage
│       └── scripts/         # Experiment scripts
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

| Experiment   | Description                      | Status       |
| ------------ | -------------------------------- | ------------ |
| Experiment 1 | Baseline Utility Assessment      | Ready to run |
| Experiment 2 | Real-world Transcript Evaluation | Planned      |
| Experiment 3 | Human Comparison                 | Planned      |
| Experiment 4 | Few-shot vs. Zero-shot Prompting | Planned      |
| Experiment 5 | Patient Communication Profiling  | Planned      |

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

- Experiment 2: Real-world transcript evaluation
- Experiment 3: Comparative assessment with human evaluators
- Experiment 4: Few-shot vs. zero-shot prompting methods
- Experiment 5: Patient communication profiling
