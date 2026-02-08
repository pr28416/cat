#!/bin/bash
# Quick status checker for running experiments

cd /Users/pranavramesh/Developer/cat

echo "=== EXPERIMENT STATUS ==="
echo "Time: $(date)"
echo ""

for exp_num in 2 3 5; do
    logfile="exp${exp_num}_output.log"
    if [ -f "$logfile" ]; then
        echo "Exp${exp_num}:"
        tail -1 "$logfile" | grep -o "Grading: *[0-9]*%|[^|]* [0-9]*/[0-9]*" || echo "  Running..."
    fi
done

echo ""
echo "=== CHECKPOINT FILES ==="
ls -lh experiments/exp*/results/processed_scores/*checkpoint*.csv 2>/dev/null | tail -10 || echo "No checkpoints yet"

