from typing import Dict, Any, Optional, Tuple
import re  # For parsing scores

from .enums import GradingCondition


def extract_transcript_from_response(llm_response_text: str) -> Optional[str]:
    """Extracts the transcript part from the LLM's response
    for synthetic transcript generation."""
    try:
        # First try to find the transcript after the "Transcript:" marker
        transcript_start_marker = "Transcript:"
        transcript_start_index = llm_response_text.find(transcript_start_marker)

        if transcript_start_index != -1:
            transcript_text = llm_response_text[
                transcript_start_index + len(transcript_start_marker) :
            ].strip()
        else:
            # If no marker found, try to find the actual dialogue content
            # Look for common dialogue patterns
            dialogue_patterns = [
                r"(?:DOCTOR|DR|PHYSICIAN|P|PATIENT|NURSE|N)[:\s].*?(?:\n(?:DOCTOR|DR|PHYSICIAN|P|PATIENT|NURSE|N)[:\s]|$)",
                r"(?:D|P)[:\s].*?(?:\n(?:D|P)[:\s]|$)",
                r"(?:Doctor|Patient)[:\s].*?(?:\n(?:Doctor|Patient)[:\s]|$)",
            ]

            for pattern in dialogue_patterns:
                matches = re.finditer(
                    pattern, llm_response_text, re.IGNORECASE | re.MULTILINE
                )
                dialogue_lines = []
                for match in matches:
                    line = match.group().strip()
                    if line and not line.startswith("Outline:"):
                        dialogue_lines.append(line)

                if dialogue_lines:
                    transcript_text = "\n".join(dialogue_lines)
                    break
            else:
                print("Error: Could not find transcript content in LLM response.")
                return None

        if not transcript_text:
            print("Error: Extracted transcript is empty.")
            return None

        return transcript_text
    except Exception as e:
        print(f"Error extracting transcript: {e}")
        return None


def parse_scores_from_response(
    response_text: str,
    grading_condition: GradingCondition,
    transcript_id: str,  # For logging/warning context
    attempt_num: int,  # For logging/warning context
) -> Dict[str, Any]:
    """Parses scores from LLM response based on the grading condition."""
    parsed_scores: Dict[str, Any] = {
        "Clarity_of_Language": None,
        "Lexical_Diversity": None,
        "Conciseness_and_Completeness": None,
        "Engagement_with_Health_Information": None,
        "Health_Literacy_Indicator": None,
        "Total_Score_Calculated": None,
        "Total_Score_Reported": None,
        "Parsing_Error": None,
    }

    try:
        if grading_condition == GradingCondition.NON_RUBRIC:
            reported_total_score = int(response_text.strip())
            if 5 <= reported_total_score <= 20:
                parsed_scores["Total_Score_Reported"] = reported_total_score
            else:
                parsed_scores["Parsing_Error"] = (
                    f"{grading_condition.value}: Reported total score {reported_total_score} out of range (5-20)."
                )

        elif grading_condition == GradingCondition.RUBRIC_BASED:
            scores_dict: Dict[str, str] = {}
            lines = response_text.strip().split("\n")
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    scores_dict[key.strip()] = value.strip()

            cat_scores: list[int] = []
            cat_keys_map: Dict[str, str] = {
                "Clarity of Language": "Clarity_of_Language",
                "Lexical Diversity": "Lexical_Diversity",
                "Conciseness and Completeness": "Conciseness_and_Completeness",
                "Engagement with Health Information": "Engagement_with_Health_Information",
                "Health Literacy Indicator": "Health_Literacy_Indicator",
            }
            valid_cats: bool = True
            for rubric_key, internal_key in cat_keys_map.items():
                if rubric_key in scores_dict:
                    try:
                        score = int(scores_dict[rubric_key])
                        if 1 <= score <= 4:
                            parsed_scores[internal_key] = score
                            cat_scores.append(score)
                        else:
                            error_msg = f"{grading_condition.value}: Category '{rubric_key}' score {score} out of range (1-4)."
                            parsed_scores["Parsing_Error"] = error_msg
                            valid_cats = False
                            break
                    except ValueError:
                        error_msg = f"{grading_condition.value}: Category '{rubric_key}' score '{scores_dict[rubric_key]}' not an integer."
                        parsed_scores["Parsing_Error"] = error_msg
                        valid_cats = False
                        break
                else:
                    error_msg = (
                        f"{grading_condition.value}: Category '{rubric_key}' missing."
                    )
                    parsed_scores["Parsing_Error"] = error_msg
                    valid_cats = False
                    break

            if valid_cats and len(cat_scores) == 5:
                calculated_total = sum(cat_scores)
                parsed_scores["Total_Score_Calculated"] = calculated_total
                if "Total Score" in scores_dict:
                    try:
                        reported_total = int(scores_dict["Total Score"])
                        parsed_scores["Total_Score_Reported"] = reported_total
                        if reported_total != calculated_total:
                            print(
                                f"Warning (ID: {transcript_id}, Att: {attempt_num}, Cond: {grading_condition.value}): Reported total ({reported_total}) != Calculated total ({calculated_total})"
                            )
                    except ValueError:
                        err_suffix = f"Total Score '{scores_dict['Total Score']}' not an integer."
                        parsed_scores["Parsing_Error"] = (
                            (parsed_scores["Parsing_Error"] + "; " + err_suffix)
                            if parsed_scores["Parsing_Error"]
                            else err_suffix
                        )
                else:
                    err_suffix = "Reported Total Score missing."
                    parsed_scores["Parsing_Error"] = (
                        (parsed_scores["Parsing_Error"] + "; " + err_suffix)
                        if parsed_scores["Parsing_Error"]
                        else err_suffix
                    )
            elif valid_cats and len(cat_scores) != 5:
                parsed_scores["Parsing_Error"] = (
                    f"{grading_condition.value}: Not all 5 category scores were parsed correctly despite valid_cats being true."
                )

    except Exception as e:
        parsed_scores["Parsing_Error"] = (
            f"General parsing error ({grading_condition.value}): {str(e)}"
        )

    return parsed_scores
