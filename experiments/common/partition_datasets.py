import os
import json
import random

# Constants from the PRD
CLEANED_DATA_DIR = "data/cleaned"
SEED_DATA_PARTITION = 12345
PARTITIONS_FILE = "experiments/dataset_partitions.json"
EXPECTED_TRANSCRIPT_COUNT = 677


def partition_datasets():
    """
    Loads transcript IDs, shuffles them, and partitions them into sets A, B, and C
    for experiments 2, 3, and 5, saving the result to a JSON file.
    """
    print("Starting dataset partitioning...")

    # 1. Get all transcript IDs from the cleaned data directory
    try:
        all_transcript_ids = [
            f for f in os.listdir(CLEANED_DATA_DIR) if f.endswith(".txt")
        ]
    except FileNotFoundError:
        print(
            f"Error: Directory not found at '{CLEANED_DATA_DIR}'. Please ensure data is in the correct location."
        )
        return

    # 2. Verify transcript count
    if len(all_transcript_ids) != EXPECTED_TRANSCRIPT_COUNT:
        print(
            f"Warning: Expected {EXPECTED_TRANSCRIPT_COUNT} transcripts, but found {len(all_transcript_ids)} in '{CLEANED_DATA_DIR}'."
        )
        # Decide if we should proceed or stop. For now, we'll proceed.
    else:
        print(f"Found {len(all_transcript_ids)} transcripts as expected.")

    # 3. Set seed and shuffle for reproducibility
    random.seed(SEED_DATA_PARTITION)
    random.shuffle(all_transcript_ids)
    print(f"Random seed set to {SEED_DATA_PARTITION} and transcript list shuffled.")

    # 4. Partition the data as per the PRD
    # Exp 2 Set (Set A): 50 transcripts
    set_a = all_transcript_ids[:50]
    # Exp 3 Set (Set B): Next 50 transcripts
    set_b = all_transcript_ids[50:100]
    # Exp 5 Set (Set C): All remaining transcripts
    set_c = all_transcript_ids[100:]

    partitions = {
        "exp2_set_A": set_a,
        "exp3_set_B": set_b,
        "exp5_set_C": set_c,
    }

    # 5. Verify partition lengths
    num_set_a = len(partitions["exp2_set_A"])
    num_set_b = len(partitions["exp3_set_B"])
    num_set_c = len(partitions["exp5_set_C"])

    print(f"Partitioned data into:")
    print(f"  - Set A (for Exp 2): {num_set_a} transcripts")
    print(f"  - Set B (for Exp 3): {num_set_b} transcripts")
    print(f"  - Set C (for Exp 5): {num_set_c} transcripts")

    assert num_set_a == 50, f"Set A should have 50 transcripts, but has {num_set_a}"
    assert num_set_b == 50, f"Set B should have 50 transcripts, but has {num_set_b}"
    assert (
        num_set_c == EXPECTED_TRANSCRIPT_COUNT - 100
    ), f"Set C should have {EXPECTED_TRANSCRIPT_COUNT - 100} transcripts, but has {num_set_c}"

    # 6. Save partitions to a JSON file
    try:
        os.makedirs(os.path.dirname(PARTITIONS_FILE), exist_ok=True)
        with open(PARTITIONS_FILE, "w") as f:
            json.dump(partitions, f, indent=4)
        print(f"Successfully saved partitions to '{PARTITIONS_FILE}'.")
    except IOError as e:
        print(f"Error saving partitions file: {e}")


if __name__ == "__main__":
    partition_datasets()
