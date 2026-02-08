#!/bin/bash
# Monitor all running experiments

clear
echo "================================================================================"
echo "EXPERIMENT MONITORING DASHBOARD"
echo "================================================================================"
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "EXPERIMENT MONITORING DASHBOARD - $(date +"%H:%M:%S")"
    echo "================================================================================"
    echo ""
    
    # Exp2
    if tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/5.txt 2>/dev/null | grep -q "Grading:"; then
        echo "📊 EXP2 (Prompt Optimization):"
        tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/5.txt | grep "Grading:"
        echo ""
    elif tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/5.txt 2>/dev/null | grep -q "COMPLETE"; then
        echo "✅ EXP2: COMPLETE"
        echo ""
    fi
    
    # Exp3
    if tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/6.txt 2>/dev/null | grep -q "Grading:"; then
        echo "📊 EXP3 (Model Comparison):"
        tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/6.txt | grep "Grading:"
        echo ""
    elif tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/6.txt 2>/dev/null | grep -q "COMPLETE"; then
        echo "✅ EXP3: COMPLETE"
        echo ""
    fi
    
    # Exp5
    if tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/7.txt 2>/dev/null | grep -q "Grading:"; then
        echo "📊 EXP5 (Final Validation):"
        tail -1 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/7.txt | grep "Grading:"
        echo ""
    elif tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/7.txt 2>/dev/null | grep -q "COMPLETE"; then
        echo "✅ EXP5: COMPLETE"
        echo ""
    fi
    
    # Checkpoint files
    echo "-------------------------------------------------------------------------------"
    echo "📁 Checkpoint Files Created:"
    echo "  Exp2: $(ls experiments/exp2_prompt_optimization/results/processed_scores/*batch*.csv 2>/dev/null | wc -l | tr -d ' ') files"
    echo "  Exp3: $(ls experiments/exp3_model_comparison/results/processed_scores/*batch*.csv 2>/dev/null | wc -l | tr -d ' ') files"
    echo "  Exp5: $(ls experiments/exp5_final_validation/results/processed_scores/*batch*.csv 2>/dev/null | wc -l | tr -d ' ') files"
    echo ""
    
    # Check if all complete
    if tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/5.txt 2>/dev/null | grep -q "COMPLETE" && \
       tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/6.txt 2>/dev/null | grep -q "COMPLETE" && \
       tail -10 /Users/pranavramesh/.cursor/projects/Users-pranavramesh-Developer-cat/terminals/7.txt 2>/dev/null | grep -q "COMPLETE"; then
        echo "================================================================================"
        echo "🎉 ALL EXPERIMENTS COMPLETE!"
        echo "================================================================================"
        break
    fi
    
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
