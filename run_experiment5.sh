#!/bin/bash

# Experiment 5: Final Tool Validation Runner
# This script runs the final validation experiment using the optimized configuration

set -e # Exit on error

echo "🚀 Starting Experiment 5: Final Tool Validation"
echo "================================================"
echo "Configuration:"
echo "  - Model: gpt-4o-2024-08-06"
echo "  - Prompt: Few-Shot (winner from Exp 2)"
echo "  - Dataset: Set C (250 transcripts for deeper analysis)"
echo "  - Target: 5,000 total assessments (20 per transcript)"
echo "  - Enhanced rate limit handling"
echo ""

# Check if we're in the right directory
if [ ! -f "experiments/dataset_partitions.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected. Attempting to activate..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Error: Virtual environment not found. Please set up the environment first."
        exit 1
    fi
fi

# Create log directory
mkdir -p logs

# Run the experiment with logging
echo "🔄 Running Experiment 5..."
echo "📝 Logs will be saved to: logs/experiment5_$(date +%Y%m%d_%H%M%S).log"

python experiments/exp5_final_validation/scripts/01_run_final_validation.py 2>&1 | tee "logs/experiment5_$(date +%Y%m%d_%H%M%S).log"

# Check if the experiment completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiment 5 completed successfully!"
    echo "🔄 Running analysis..."

    # Run the analysis
    python experiments/exp5_final_validation/scripts/02_analyze_final_validation.py 2>&1 | tee "logs/experiment5_analysis_$(date +%Y%m%d_%H%M%S).log"

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Experiment 5 Analysis completed successfully!"
        echo "📊 Check the results in: experiments/exp5_final_validation/results/"
        echo ""
        echo "📋 Next steps:"
        echo "  1. Review the analysis report"
        echo "  2. Check if PRD success metrics were achieved"
        echo "  3. Proceed with manuscript preparation if benchmarks are met"
    else
        echo "❌ Analysis failed. Check the logs for details."
        exit 1
    fi
else
    echo "❌ Experiment 5 failed. Check the logs for details."
    exit 1
fi
