"""
Model Constants for CHAT Experiments

This module defines consistent model names across all experiments to ensure
paper claims match code implementation.
"""

# Default model for the final CHAT system
DEFAULT_FINAL_MODEL = "gpt-5-nano"

# GPT-5 family models
GPT5_MODELS = [
    "gpt-5-nano",  # Default, optimal for cost/performance
    "gpt-5-mini",  # Robustness comparison
]

# Legacy models (kept only for comparison experiments, not as final configuration)
LEGACY_COMPARISON_MODELS = [
    "gpt-4o-2024-08-06",      # Previous best general GPT-4o
    "gpt-4o-mini-2024-07-18", # Previous cost-efficient model
    "gpt-4.1-2025-04-14",     # GPT-4.1 baseline
    "o3-2025-04-16",          # Reasoning model
    "o3-mini-2025-01-31",     # Smaller reasoning model
]

# All models used across experiments
ALL_MODELS = GPT5_MODELS + LEGACY_COMPARISON_MODELS

# Temperature settings by experiment type
TEMPERATURES = {
    "optimization": 0.1,      # Exp2, Exp3 - consistency priority
    "final_validation": 0.3,  # Exp5 - balance consistency and natural language
    "robustness": 0.3,        # Exp7 - match Exp5 settings
}

# Worker settings for multithreading
DEFAULT_MAX_WORKERS = 8
MIN_WORKERS = 6
MAX_WORKERS = 8


def get_max_workers():
    """Get max workers from environment or return default, clamped to [6, 8]."""
    import os
    try:
        workers = int(os.getenv("CHAT_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))
        return max(MIN_WORKERS, min(MAX_WORKERS, workers))
    except ValueError:
        return DEFAULT_MAX_WORKERS

