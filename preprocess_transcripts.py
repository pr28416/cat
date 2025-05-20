import os
import shutil
import re  # Added for regex operations

# Define paths
NATURE_RAW_DIR = "data/raw/Nature Transcripts/"
DOVA_RAW_DIR = (
    "data/raw/DoVA Transcripts/Transcripts/"  # Adjusted to the correct subfolder
)
CLEANED_DIR = "data/cleaned/"

# Clear out the cleaned directory first
if os.path.exists(CLEANED_DIR):
    print(f"Clearing existing contents from {CLEANED_DIR}...")
    for item in os.listdir(CLEANED_DIR):
        item_path = os.path.join(CLEANED_DIR, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")
    print(f"Finished clearing {CLEANED_DIR}.")

# Ensure cleaned directory exists
os.makedirs(CLEANED_DIR, exist_ok=True)

processed_files_count = 0
skipped_files_count = 0

# Define speaker standardization rules
# Each tuple: (regex_pattern_str, standard_speaker_tag_str ("DOCTOR" or "PATIENT"), rule_applies_to_prefix ("DOVA_", "NATURE_", "COMMON_"))
# Patterns are matched at the beginning of the line, case-insensitive.
SPEAKER_RULES = [
    # DoVA specific rules: Match full speaker ID patterns, including ordinals.
    # Lines like "PATIENT?" on their own will not be treated as a new PATIENT speaker line by these rules,
    # allowing them to be consumed as dialogue by a preceding DOCTOR speaker line with an empty payload.
    (
        r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)?\s*(?:DOCTOR|DR)\s+[A-Z0-9]+\b\s*:?",
        "DOCTOR",
        "DOVA_",
    ),
    (
        r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH)?\s*(?:PATIENT|PT)\s+[A-Z0-9]+\b\s*:?",
        "PATIENT",
        "DOVA_",
    ),
    # Nature specific
    (r"^D\s*:", "DOCTOR", "NATURE_"),
    (r"^P\s*:", "PATIENT", "NATURE_"),
]


def standardize_transcript(input_path, output_path, transcript_id_prefix):
    """
    Reads a transcript file, standardizes its content by cleaning headers/footers
    and applying a uniform dialogue format, and saves it to the output path.
    """
    global processed_files_count, skipped_files_count
    try:
        filename = os.path.basename(input_path)
        transcript_id = f"{transcript_id_prefix}{filename}"
        output_file_path = os.path.join(output_path, transcript_id)

        with open(input_path, "r", encoding="utf-8", errors="ignore") as infile:
            original_lines = infile.readlines()

        # 1. Initial Header/Footer Cleaning (specific to data source)
        dialogue_lines_after_header_removal = []
        if transcript_id_prefix == "DOVA_":
            dialogue_start_index = 0
            found_dialogue = False
            speaker_prefixes = ["DOCTOR ", "PATIENT ", "DR ", "PT ", "NURSE "]
            for i, line in enumerate(original_lines[:10]):
                if any(
                    line.strip().upper().startswith(prefix)
                    for prefix in speaker_prefixes
                ):
                    if (
                        i > 1
                        and original_lines[i - 1].strip() == ""
                        and "ENCOUNTER:" in original_lines[i - 2].upper()
                    ):
                        dialogue_start_index = i
                        found_dialogue = True
                        break
                    elif any(kw in line.upper() for kw in ["DOCTOR", "PATIENT"]):
                        dialogue_start_index = i
                        found_dialogue = True
                        break
            if not found_dialogue:
                for i, line in enumerate(original_lines):
                    stripped_line = line.strip()
                    if (
                        stripped_line
                        and (":" in stripped_line or len(stripped_line.split()) < 7)
                        and (
                            not any(
                                kw in stripped_line.upper()
                                for kw in [
                                    "DATE OF ENCOUNTER",
                                    "TIME OF ENCOUNTER",
                                    transcript_id_prefix.strip("_"),
                                ]
                            )
                        )
                    ):
                        dialogue_start_index = i
                        found_dialogue = True
                        break
                if not found_dialogue:
                    dialogue_start_index = 3
                    # print(f"Debug: DoVA Fallback for {filename}. Skipping first {dialogue_start_index} lines.")

            dialogue_lines_after_header_removal = original_lines[dialogue_start_index:]
        elif transcript_id_prefix == "NATURE_":
            # Remove only leading blank lines for Nature, then content stripping is per line
            start_index = 0
            for i, line_content in enumerate(original_lines):
                if line_content.strip():
                    start_index = i
                    break
            dialogue_lines_after_header_removal = original_lines[start_index:]
        else:
            dialogue_lines_after_header_removal = original_lines

        # 2. Standardize Dialogue Format (@@@ replacement, speaker tags)
        final_standardized_lines = []
        i = 0
        num_dialogue_lines = len(dialogue_lines_after_header_removal)

        while i < num_dialogue_lines:
            current_line_content = dialogue_lines_after_header_removal[i]
            processed_line = current_line_content.strip()
            i += 1  # Increment for the current line

            if not processed_line:  # Skip lines that are now empty
                continue

            # Replace @@@ globally in the line with [REDACTED]
            processed_line = processed_line.replace("@@@", "[REDACTED]")

            text_payload = processed_line
            is_speaker_line = False
            current_speaker_tag = ""

            for pattern_str, standard_tag, rule_prefix in SPEAKER_RULES:
                if rule_prefix == transcript_id_prefix or rule_prefix == "COMMON_":
                    match = re.match(pattern_str, processed_line, re.IGNORECASE)
                    if match:
                        text_payload = processed_line[match.end(0) :].strip()
                        current_speaker_tag = standard_tag
                        is_speaker_line = True
                        break

            if is_speaker_line:
                # If it's a speaker line AND the text payload is empty, try to look ahead
                if not text_payload:
                    # Peek at the next non-empty line
                    j = i  # Start peeking from the line after current one
                    while j < num_dialogue_lines:
                        next_line_content = dialogue_lines_after_header_removal[
                            j
                        ].strip()
                        if next_line_content:  # Found a non-empty line
                            # Process this peeked line for @@@ (now [REDACTED])
                            peek_processed_line = next_line_content.replace(
                                "@@@", "[REDACTED]"
                            )

                            # Check if this peeked line is also a speaker line
                            is_peeked_line_speaker = False
                            for p_pattern_str, _, p_rule_prefix in SPEAKER_RULES:
                                if (
                                    p_rule_prefix == transcript_id_prefix
                                    or p_rule_prefix == "COMMON_"
                                ):
                                    if re.match(
                                        p_pattern_str,
                                        peek_processed_line,
                                        re.IGNORECASE,
                                    ):
                                        is_peeked_line_speaker = True
                                        break

                            if not is_peeked_line_speaker:
                                # It's a content line for the current speaker, merge it
                                text_payload = peek_processed_line
                                i = (
                                    j + 1
                                )  # Advance main loop counter past this consumed line
                            break  # Stop peeking whether it was merged or not
                        j += 1  # Continue peeking if current peek was empty

                # Construct the speaker line
                final_line = (
                    f"{current_speaker_tag}: {text_payload}"
                    if text_payload
                    else f"{current_speaker_tag}:"
                )
                final_standardized_lines.append(final_line)
            else:
                # Not a speaker line, just add the processed (stripped, @@@ replaced) line
                final_standardized_lines.append(processed_line)

        cleaned_content = "\n".join(final_standardized_lines)

        if not cleaned_content.strip():  # Check if overall content is empty
            print(
                f"Warning: File {input_path} resulted in empty content after cleaning. Skipping."
            )
            skipped_files_count += 1
            return False

        with open(output_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(cleaned_content)

        processed_files_count += 1
        return True
    except Exception as e:
        print(f"Error processing file {input_path}: {e}")
        skipped_files_count += 1
        return False


# Process Nature Transcripts
print(f"Processing Nature transcripts from: {NATURE_RAW_DIR}")
if os.path.exists(NATURE_RAW_DIR):
    for item in os.listdir(NATURE_RAW_DIR):
        item_path = os.path.join(NATURE_RAW_DIR, item)
        if os.path.isfile(item_path) and item.endswith(".txt"):
            standardize_transcript(item_path, CLEANED_DIR, "NATURE_")
else:
    print(f"Warning: Directory not found - {NATURE_RAW_DIR}")


# Process DoVA Transcripts
print(f"Processing DoVA transcripts from: {DOVA_RAW_DIR}")
if os.path.exists(DOVA_RAW_DIR):
    for item in os.listdir(DOVA_RAW_DIR):
        item_path = os.path.join(DOVA_RAW_DIR, item)
        if os.path.isfile(item_path) and item.endswith(".txt"):
            standardize_transcript(item_path, CLEANED_DIR, "DOVA_")
else:
    print(f"Warning: Directory not found - {DOVA_RAW_DIR}")

print(f"\n--- Preprocessing Complete ---")
print(f"Total files processed: {processed_files_count}")
print(f"Total files skipped due to errors: {skipped_files_count}")
print(f"Cleaned files are located in: {CLEANED_DIR}")

# Verification: List some files from the cleaned directory
if processed_files_count > 0:
    print("\nSample of cleaned files:")
    cleaned_files_sample = os.listdir(CLEANED_DIR)[:5]
    for fname in cleaned_files_sample:
        print(f"- {fname}")


def evaluate_cleaned_files(cleaned_dir):
    """
    Evaluates the cleaned transcript files against a set of defined constraints.
    """
    print("\n--- Starting Evaluation of Cleaned Files ---")
    issues_found = {}
    files_checked = 0
    passed_files_count = 0

    # Raw speaker patterns to check for absence (examples)
    old_speaker_patterns = [
        re.compile(r"^D\s*:\s*", re.IGNORECASE),
        re.compile(r"^P\s*:\s*", re.IGNORECASE),
        re.compile(r"^(DOCTOR|DR)\s+[A-Z0-9]+\b", re.IGNORECASE),
        re.compile(r"^(PATIENT|PT)\s+[A-Z0-9]+\b", re.IGNORECASE),
    ]

    for filename in os.listdir(cleaned_dir):
        if not filename.endswith(".txt"):
            continue

        files_checked += 1
        file_path = os.path.join(cleaned_dir, filename)
        file_issues = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                file_issues.append("File is empty.")

            has_dialogue_content = False
            consecutive_blank_lines = 0

            for i, line_content in enumerate(lines):
                if line_content.rstrip("\n") != line_content.rstrip():
                    file_issues.append(
                        f"Line {i+1}: Contains trailing whitespace beyond newline: '{line_content.rstrip('\n')[:50]}...'"
                    )

                stripped_line = line_content.strip()

                if not stripped_line:
                    consecutive_blank_lines += 1
                    if consecutive_blank_lines > 1:
                        file_issues.append(
                            f"Line {i+1}: Multiple consecutive blank lines (>{consecutive_blank_lines-1})."
                        )
                    continue
                else:
                    consecutive_blank_lines = 0
                    has_dialogue_content = True

                if "@@@" in stripped_line:
                    file_issues.append(f"Line {i+1}: Contains raw '@@@'.")

                if "[INAUDIBLE]" in stripped_line:
                    file_issues.append(
                        f"Line {i+1}: Contains '[INAUDIBLE]' instead of '[REDACTED]'."
                    )

                # Check for old speaker patterns not correctly handled
                # This check should ideally not find anything if main logic is correct
                is_actually_valid_dialogue_containing_pattern = False
                if stripped_line.startswith(("DOCTOR: ", "PATIENT: ")):
                    # Check if the old pattern appears *after* a valid new tag (e.g. DOCTOR: P: ...)
                    temp_payload = (
                        stripped_line.split(": ", 1)[1] if ": " in stripped_line else ""
                    )
                    for pattern in old_speaker_patterns:
                        if pattern.match(temp_payload):
                            # This means an old pattern might be part of the dialogue text itself.
                            # This is permissible, so we don't flag it as an error here.
                            # Example: PATIENT: Doctor, D: my old label, told me... - this is fine.
                            is_actually_valid_dialogue_containing_pattern = True
                            break

                if not is_actually_valid_dialogue_containing_pattern:
                    for pattern in old_speaker_patterns:
                        if pattern.match(stripped_line):
                            file_issues.append(
                                f"Line {i+1}: Starts with an old/unconverted speaker pattern: '{stripped_line[:50]}...'"
                            )
                            break  # Found one, no need to check others for this line

                # Constraint: Lines starting with DOCTOR/PATIENT should be formatted correctly
                if stripped_line.startswith("DOCTOR:"):
                    if not stripped_line.startswith("DOCTOR: ") and len(
                        stripped_line
                    ) > len("DOCTOR:"):
                        file_issues.append(
                            f"Line {i+1}: 'DOCTOR:' not followed by space: '{stripped_line[:50]}...'"
                        )
                elif stripped_line.startswith("PATIENT:"):
                    if not stripped_line.startswith("PATIENT: ") and len(
                        stripped_line
                    ) > len("PATIENT:"):
                        file_issues.append(
                            f"Line {i+1}: 'PATIENT:' not followed by space: '{stripped_line[:50]}...'"
                        )

            if not has_dialogue_content and lines:
                file_issues.append("File contains only blank lines after stripping.")

        except Exception as e:
            file_issues.append(f"Error reading or processing file: {e}")

        if file_issues:
            issues_found[filename] = file_issues
        else:
            passed_files_count += 1

    print("\nEvaluation Summary:")
    print(f"Total files checked: {files_checked}")
    print(f"Files passed all checks: {passed_files_count}")
    print(f"Files with issues: {len(issues_found)}")

    if issues_found:
        print("\nDetails of files with issues:")
        for fname, f_issues in issues_found.items():
            print(f"\n--- File: {fname} ---")
            for issue in f_issues:
                print(f"  - {issue}")
    print("\n--- Evaluation Complete ---")


# --- Call evaluation right before the script ends ---
# The main processing loops are at the global level and will run before this.
if processed_files_count > 0 or (
    os.path.exists(CLEANED_DIR)
    and any(fname.endswith(".txt") for fname in os.listdir(CLEANED_DIR))
):
    evaluate_cleaned_files(CLEANED_DIR)
else:
    print(
        "\nSkipping evaluation as no files were processed or cleaned directory is empty/does not exist or contains no .txt files."
    )
