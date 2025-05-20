#!/bin/bash
# Run Experiment 1: Baseline Utility Assessment

echo "===== Starting Experiment 1: Baseline Utility Assessment ====="

# Get the absolute path to the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set current directory to project root (containing the script)
cd "$SCRIPT_DIR"

# Ensure the Python environment is active
if [ -d ".venv" ]; then
    echo "Activating Python virtual environment..."
    source .venv/bin/activate
fi

# Check for environment file
if [ ! -f ".env.local" ]; then
    echo "Error: .env.local file not found."
    echo "Please create a .env.local file with your OpenAI API key:"
    echo "OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "OPENAI_API_KEY" .env.local && ! grep -q "OPENAI_API_KEYS" .env.local; then
    echo "Error: No OpenAI API key found in .env.local."
    echo "Please add your OpenAI API key to .env.local:"
    echo "OPENAI_API_KEY=your_api_key_here"
    exit 1
fi

# Create necessary directories
mkdir -p data/synthetic experiments/exp1_baseline_utility/results/analysis experiments/exp1_baseline_utility/results/processed_scores

# Run experiment from project root
echo "Step 1: Generating synthetic transcripts..."
python -m experiments.exp1_baseline_utility.scripts.01_generate_synthetic_transcripts
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate synthetic transcripts."
    exit 1
fi

# Check if synthetic transcripts were generated
if [ ! -f "data/synthetic/exp1_synthetic_transcripts.csv" ]; then
    echo "Error: Synthetic transcripts file not created."
    exit 1
fi

# Step 2: Grade the synthetic transcripts
echo "Step 2: Grading synthetic transcripts..."
python -m experiments.exp1_baseline_utility.scripts.02_run_grading
if [ $? -ne 0 ]; then
    echo "Error: Failed to grade synthetic transcripts."
    exit 1
fi

# Step 3: Analyze the results
echo "Step 3: Analyzing results..."
python -m experiments.exp1_baseline_utility.scripts.03_analyze_results
if [ $? -ne 0 ]; then
    echo "Error: Failed to analyze results."
    exit 1
fi

echo "===== Experiment 1 completed ====="
