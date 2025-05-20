import os
from typing import Any
import matplotlib.pyplot as plt


def load_text_file(filepath: str) -> str:
    """Loads text content from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def ensure_dir_exists(directory: str) -> None:
    """Ensures that the specified directory exists."""
    os.makedirs(directory, exist_ok=True)


def save_plot(fig: plt.Figure, filename: str, analysis_output_dir: str) -> None:
    """Saves the given matplotlib figure to the specified filename in the analysis output directory."""
    path = os.path.join(analysis_output_dir, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close(fig)  # Close the figure to free memory
