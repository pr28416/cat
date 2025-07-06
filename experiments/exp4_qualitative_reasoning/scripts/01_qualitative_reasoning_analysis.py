#!/usr/bin/env python3
"""
Experiment 4: Qualitative Reasoning Analysis

This script analyzes the reasoning patterns of different LLMs when applying
the Patient Health Communication Rubric v5.0, focusing on:
1. Evidence citation patterns
2. Rubric alignment quality
3. Reasoning depth and specificity
4. Identification of model-specific strengths/weaknesses
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import defaultdict, Counter
from datetime import datetime
import json
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- Constants ---
EXPERIMENT_ID = "EXP4_ReasoningAnalysis"
EXP3_RESULTS_FILE = "exp3_modelcomparison_grading_results_curated_models.csv"
ALPHA = 0.05
RANDOM_SEED = 42

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EXP3_RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/processed_scores"
)
TRANSCRIPTS_DIR = os.path.join(PROJECT_ROOT, "data/cleaned")
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp4_qualitative_reasoning/results"
)
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
PROCESSED_DIR = os.path.join(RESULTS_DIR, "processed_scores")

# Ensure directories exist
for dir_path in [RESULTS_DIR, ANALYSIS_DIR, PROCESSED_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Rubric Categories ---
RUBRIC_CATEGORIES = [
    "Clarity_of_Language",
    "Lexical_Diversity",
    "Conciseness_and_Completeness",
    "Engagement_with_Health_Information",
    "Health_Literacy_Indicator",
]


class ReasoningAnalyzer:
    """Main class for analyzing LLM reasoning patterns."""

    def __init__(self):
        self.df = None
        self.transcripts_selected = []
        self.coding_results = []

    def load_data(self):
        """Load Experiment 3 results and select diverse transcripts."""
        print("=" * 60)
        print("🔍 LOADING EXPERIMENT 3 DATA")
        print("=" * 60)

        exp3_path = os.path.join(EXP3_RESULTS_DIR, EXP3_RESULTS_FILE)
        if not os.path.exists(exp3_path):
            raise FileNotFoundError(f"Experiment 3 results not found: {exp3_path}")

        self.df = pd.read_csv(exp3_path)

        # Filter out parsing errors
        self.df = self.df[self.df["Parsing_Error"].isna()]

        print(f"📊 Dataset Overview:")
        print(f"  Total records: {len(self.df)}")
        print(f"  Models: {', '.join(self.df['model_name'].unique())}")
        print(f"  Transcripts: {self.df['transcript_id'].nunique()}")
        print(f"  Success rate: 100% (pre-filtered)")

        return self.df

    def select_diverse_transcripts(self, n_transcripts=15):
        """Select diverse transcripts based on consistency patterns from Exp 3."""
        print("\n" + "=" * 60)
        print("🎯 SELECTING DIVERSE TRANSCRIPTS")
        print("=" * 60)

        # Calculate consistency metrics per transcript
        consistency_stats = []
        for transcript_id in self.df["transcript_id"].unique():
            transcript_data = self.df[self.df["transcript_id"] == transcript_id]

            # Calculate STDEV for each model
            model_stdevs = []
            for model in transcript_data["model_name"].unique():
                model_data = transcript_data[transcript_data["model_name"] == model]
                scores = model_data["Total_Score_Calculated"]
                if len(scores) > 1:
                    model_stdevs.append(scores.std())

            if model_stdevs:
                consistency_stats.append(
                    {
                        "transcript_id": transcript_id,
                        "mean_stdev": np.mean(model_stdevs),
                        "max_stdev": np.max(model_stdevs),
                        "min_stdev": np.min(model_stdevs),
                        "stdev_range": np.max(model_stdevs) - np.min(model_stdevs),
                    }
                )

        consistency_df = pd.DataFrame(consistency_stats)

        # Select transcripts with diverse consistency patterns
        # Sort by mean_stdev to get range from most to least consistent
        consistency_df = consistency_df.sort_values("mean_stdev")

        # Select transcripts: 5 most consistent, 5 middle, 5 least consistent
        n_per_group = n_transcripts // 3
        selected_transcripts = []

        # Most consistent (low STDEV)
        selected_transcripts.extend(
            consistency_df.head(n_per_group)["transcript_id"].tolist()
        )

        # Middle consistency
        mid_start = len(consistency_df) // 2 - n_per_group // 2
        selected_transcripts.extend(
            consistency_df.iloc[mid_start : mid_start + n_per_group][
                "transcript_id"
            ].tolist()
        )

        # Least consistent (high STDEV)
        selected_transcripts.extend(
            consistency_df.tail(n_per_group)["transcript_id"].tolist()
        )

        # Fill remaining slots if needed
        while len(selected_transcripts) < n_transcripts:
            remaining = consistency_df[
                ~consistency_df["transcript_id"].isin(selected_transcripts)
            ]
            if len(remaining) > 0:
                selected_transcripts.append(remaining.iloc[0]["transcript_id"])
            else:
                break

        self.transcripts_selected = selected_transcripts[:n_transcripts]

        print(f"📋 Selected {len(self.transcripts_selected)} transcripts:")
        for i, tid in enumerate(self.transcripts_selected, 1):
            stats = consistency_df[consistency_df["transcript_id"] == tid].iloc[0]
            print(
                f"  {i:2d}. {tid}: mean_stdev={stats['mean_stdev']:.3f}, range={stats['stdev_range']:.3f}"
            )

        return self.transcripts_selected

    def load_transcript_content(self, transcript_id):
        """Load the actual transcript content."""
        transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_id)
        if os.path.exists(transcript_path):
            with open(transcript_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def extract_reasoning_segments(self, raw_response):
        """Extract reasoning segments from model responses."""
        # Split response into scores and rationale
        lines = raw_response.split("\n")

        # Find the rationale section (usually after "Rationale:" or similar)
        rationale_start = -1
        for i, line in enumerate(lines):
            if any(
                keyword in line.lower()
                for keyword in ["rationale", "reasoning", "explanation"]
            ):
                rationale_start = i
                break

        if rationale_start == -1:
            # Look for the rationale after the scores
            score_end = -1
            for i, line in enumerate(lines):
                if "Total Score:" in line:
                    score_end = i
                    break
            if score_end != -1:
                rationale_start = score_end + 1

        # Extract rationale text
        if rationale_start != -1 and rationale_start < len(lines):
            rationale_lines = lines[rationale_start:]
            rationale = "\n".join(rationale_lines).strip()

            # Clean up rationale markers
            rationale = re.sub(r"^\*?\*?[Rr]ationale:?\*?\*?", "", rationale).strip()
            rationale = re.sub(r"^\*?\*?[Ee]xplanation:?\*?\*?", "", rationale).strip()

            return rationale

        return raw_response  # Return full response if no clear rationale section

    def code_reasoning_quality(self, reasoning_text, transcript_content=None):
        """Apply coding scheme to reasoning text."""
        coding_result = {}

        # 1. Evidence Type Analysis
        evidence_types = []

        # Direct quotes (text in quotes)
        if re.search(r'["\'].*?["\']', reasoning_text):
            evidence_types.append("Direct Quote")

        # Paraphrasing indicators
        if any(
            indicator in reasoning_text.lower()
            for indicator in ["such as", "like", "including", "for example", "e.g."]
        ):
            evidence_types.append("Paraphrase")

        # General statements without specific evidence
        if not evidence_types:
            evidence_types.append("General Statement")

        coding_result["evidence_types"] = evidence_types

        # 2. Rubric Alignment Quality
        rubric_alignment = "Loose"  # Default

        # Check for explicit rubric category mentions
        category_mentions = 0
        for category in RUBRIC_CATEGORIES:
            category_clean = category.replace("_", " ").lower()
            if category_clean in reasoning_text.lower():
                category_mentions += 1

        if category_mentions >= 3:
            rubric_alignment = "Clear"
        elif category_mentions >= 1:
            rubric_alignment = "Moderate"
        else:
            rubric_alignment = "Loose"

        coding_result["rubric_alignment"] = rubric_alignment

        # 3. Reasoning Depth
        depth_indicators = [
            "because",
            "since",
            "due to",
            "as a result",
            "therefore",
            "however",
            "although",
            "while",
            "despite",
            "in contrast",
        ]

        depth_score = sum(
            1 for indicator in depth_indicators if indicator in reasoning_text.lower()
        )

        if depth_score >= 3:
            reasoning_depth = "In-depth"
        elif depth_score >= 1:
            reasoning_depth = "Moderate"
        else:
            reasoning_depth = "Superficial"

        coding_result["reasoning_depth"] = reasoning_depth

        # 4. Specificity Analysis
        specificity_indicators = [
            "specific",
            "particular",
            "exact",
            "precise",
            "detailed",
            "clearly",
            "explicitly",
            "demonstrates",
            "shows",
            "indicates",
        ]

        specificity_score = sum(
            1
            for indicator in specificity_indicators
            if indicator in reasoning_text.lower()
        )

        if specificity_score >= 2:
            specificity = "Specific"
        elif specificity_score >= 1:
            specificity = "Moderate"
        else:
            specificity = "Generic"

        coding_result["specificity"] = specificity

        # 5. Balanced Assessment
        positive_indicators = ["good", "clear", "appropriate", "effective", "strong"]
        negative_indicators = ["poor", "unclear", "limited", "weak", "lacking"]

        pos_count = sum(
            1
            for indicator in positive_indicators
            if indicator in reasoning_text.lower()
        )
        neg_count = sum(
            1
            for indicator in negative_indicators
            if indicator in reasoning_text.lower()
        )

        if pos_count > 0 and neg_count > 0:
            balance = "Balanced"
        elif pos_count > neg_count:
            balance = "Positive Focus"
        elif neg_count > pos_count:
            balance = "Negative Focus"
        else:
            balance = "Neutral"

        coding_result["balance"] = balance

        # 6. Length and Structure
        word_count = len(reasoning_text.split())
        sentence_count = len(re.split(r"[.!?]+", reasoning_text))

        coding_result["word_count"] = word_count
        coding_result["sentence_count"] = sentence_count

        # Structure indicators
        has_bullets = bool(re.search(r"[-*•]", reasoning_text))
        has_formatting = bool(re.search(r"\*\*.*?\*\*", reasoning_text))

        coding_result["has_bullets"] = has_bullets
        coding_result["has_formatting"] = has_formatting

        return coding_result

    def analyze_all_reasoning(self):
        """Analyze reasoning for all selected transcripts and models."""
        print("\n" + "=" * 60)
        print("🔍 ANALYZING REASONING PATTERNS")
        print("=" * 60)

        results = []

        for transcript_id in self.transcripts_selected:
            print(f"\n📋 Processing {transcript_id}...")

            # Load transcript content
            transcript_content = self.load_transcript_content(transcript_id)

            # Get all responses for this transcript
            transcript_data = self.df[self.df["transcript_id"] == transcript_id]

            for model in transcript_data["model_name"].unique():
                model_data = transcript_data[transcript_data["model_name"] == model]

                # Analyze first response (they should be similar due to low temperature)
                sample_response = model_data.iloc[0]
                raw_response = sample_response["raw_response"]

                # Extract reasoning
                reasoning_text = self.extract_reasoning_segments(raw_response)

                # Apply coding scheme
                coding_result = self.code_reasoning_quality(
                    reasoning_text, transcript_content
                )

                # Combine with metadata
                result = {
                    "transcript_id": transcript_id,
                    "model_name": model,
                    "reasoning_text": reasoning_text,
                    "reasoning_length": len(reasoning_text),
                    **coding_result,
                }

                results.append(result)

        self.coding_results = pd.DataFrame(results)

        print(f"\n✅ Analyzed {len(results)} reasoning samples")
        print(
            f"   ({len(self.transcripts_selected)} transcripts × {len(self.df['model_name'].unique())} models)"
        )

        return self.coding_results

    def statistical_analysis(self):
        """Perform statistical analysis of coding results."""
        print("\n" + "=" * 60)
        print("📊 STATISTICAL ANALYSIS")
        print("=" * 60)

        models = self.coding_results["model_name"].unique()

        # 1. Evidence Type Analysis
        print("\n1️⃣ Evidence Type Patterns:")
        evidence_summary = []
        for model in models:
            model_data = self.coding_results[self.coding_results["model_name"] == model]

            # Count evidence types (can be multiple per response)
            evidence_counts = Counter()
            for evidence_list in model_data["evidence_types"]:
                for evidence_type in evidence_list:
                    evidence_counts[evidence_type] += 1

            total_responses = len(model_data)
            evidence_summary.append(
                {
                    "model": model,
                    "direct_quote_pct": evidence_counts["Direct Quote"]
                    / total_responses
                    * 100,
                    "paraphrase_pct": evidence_counts["Paraphrase"]
                    / total_responses
                    * 100,
                    "general_statement_pct": evidence_counts["General Statement"]
                    / total_responses
                    * 100,
                }
            )

        evidence_df = pd.DataFrame(evidence_summary)
        print(evidence_df.round(1))

        # 2. Rubric Alignment Analysis
        print("\n2️⃣ Rubric Alignment Quality:")
        alignment_crosstab = (
            pd.crosstab(
                self.coding_results["model_name"],
                self.coding_results["rubric_alignment"],
                normalize="index",
            )
            * 100
        )
        print(alignment_crosstab.round(1))

        # 3. Reasoning Depth Analysis
        print("\n3️⃣ Reasoning Depth:")
        depth_crosstab = (
            pd.crosstab(
                self.coding_results["model_name"],
                self.coding_results["reasoning_depth"],
                normalize="index",
            )
            * 100
        )
        print(depth_crosstab.round(1))

        # 4. Length and Structure Analysis
        print("\n4️⃣ Length and Structure:")
        structure_summary = (
            self.coding_results.groupby("model_name")
            .agg(
                {
                    "word_count": ["mean", "std"],
                    "sentence_count": ["mean", "std"],
                    "has_bullets": "mean",
                    "has_formatting": "mean",
                }
            )
            .round(2)
        )
        print(structure_summary)

        # Save detailed results
        summary_path = os.path.join(PROCESSED_DIR, "reasoning_analysis_summary.csv")
        self.coding_results.to_csv(summary_path, index=False)
        print(f"\n💾 Detailed results saved to: {summary_path}")

        return {
            "evidence_df": evidence_df,
            "alignment_crosstab": alignment_crosstab,
            "depth_crosstab": depth_crosstab,
            "structure_summary": structure_summary,
        }

    def create_visualizations(self, stats_dict):
        """Create visualizations for the analysis."""
        print("\n" + "=" * 60)
        print("📈 CREATING VISUALIZATIONS")
        print("=" * 60)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Rubric Alignment Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            stats_dict["alignment_crosstab"],
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            cbar_kws={"label": "Percentage"},
        )
        plt.title("Rubric Alignment Quality by Model", fontsize=14, fontweight="bold")
        plt.xlabel("Alignment Quality")
        plt.ylabel("Model")
        plt.tight_layout()

        alignment_plot_path = os.path.join(
            ANALYSIS_DIR, "exp4_rubric_alignment_heatmap.png"
        )
        plt.savefig(alignment_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved alignment heatmap: {alignment_plot_path}")

        # 2. Reasoning Depth Comparison
        plt.figure(figsize=(12, 6))
        depth_data = stats_dict["depth_crosstab"].reset_index()
        depth_melted = pd.melt(
            depth_data,
            id_vars=["model_name"],
            var_name="depth_level",
            value_name="percentage",
        )

        sns.barplot(
            data=depth_melted, x="model_name", y="percentage", hue="depth_level"
        )
        plt.title("Reasoning Depth by Model", fontsize=14, fontweight="bold")
        plt.xlabel("Model")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Reasoning Depth")
        plt.tight_layout()

        depth_plot_path = os.path.join(
            ANALYSIS_DIR, "exp4_reasoning_depth_comparison.png"
        )
        plt.savefig(depth_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved depth comparison: {depth_plot_path}")

        # 3. Response Length Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.coding_results, x="model_name", y="word_count")
        plt.title(
            "Response Length Distribution by Model", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Model")
        plt.ylabel("Word Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        length_plot_path = os.path.join(
            ANALYSIS_DIR, "exp4_response_length_distribution.png"
        )
        plt.savefig(length_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved length distribution: {length_plot_path}")

        # 4. Evidence Type Usage
        plt.figure(figsize=(10, 6))
        evidence_df = stats_dict["evidence_df"].set_index("model")
        evidence_df[
            ["direct_quote_pct", "paraphrase_pct", "general_statement_pct"]
        ].plot(kind="bar", stacked=True, ax=plt.gca())
        plt.title("Evidence Type Usage by Model", fontsize=14, fontweight="bold")
        plt.xlabel("Model")
        plt.ylabel("Percentage")
        plt.legend(["Direct Quote", "Paraphrase", "General Statement"])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        evidence_plot_path = os.path.join(ANALYSIS_DIR, "exp4_evidence_type_usage.png")
        plt.savefig(evidence_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved evidence usage plot: {evidence_plot_path}")

    def generate_qualitative_examples(self):
        """Generate qualitative examples showing reasoning differences."""
        print("\n" + "=" * 60)
        print("📝 GENERATING QUALITATIVE EXAMPLES")
        print("=" * 60)

        examples = []

        # Select one transcript for detailed comparison
        sample_transcript = self.transcripts_selected[0]
        sample_data = self.coding_results[
            self.coding_results["transcript_id"] == sample_transcript
        ]

        print(f"📋 Sample Analysis for {sample_transcript}:")
        print("=" * 50)

        for _, row in sample_data.iterrows():
            model = row["model_name"]
            reasoning = (
                row["reasoning_text"][:300] + "..."
                if len(row["reasoning_text"]) > 300
                else row["reasoning_text"]
            )

            example = {
                "transcript_id": sample_transcript,
                "model_name": model,
                "reasoning_sample": reasoning,
                "evidence_types": row["evidence_types"],
                "rubric_alignment": row["rubric_alignment"],
                "reasoning_depth": row["reasoning_depth"],
                "word_count": row["word_count"],
            }

            examples.append(example)

            print(f"\n{model}:")
            print(f"  Alignment: {row['rubric_alignment']}")
            print(f"  Depth: {row['reasoning_depth']}")
            print(f"  Words: {row['word_count']}")
            print(f"  Sample: {reasoning}")

        # Save examples
        examples_path = os.path.join(ANALYSIS_DIR, "qualitative_examples.json")
        with open(examples_path, "w") as f:
            json.dump(examples, f, indent=2)

        print(f"\n💾 Qualitative examples saved to: {examples_path}")
        return examples

    def save_comprehensive_report(self, stats_dict):
        """Save a comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("💾 SAVING COMPREHENSIVE REPORT")
        print("=" * 60)

        report_path = os.path.join(ANALYSIS_DIR, "exp4_reasoning_analysis_report.txt")

        with open(report_path, "w") as f:
            f.write("EXPERIMENT 4: QUALITATIVE REASONING ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("METHODOLOGY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Transcripts analyzed: {len(self.transcripts_selected)}\n")
            f.write(
                f"Models compared: {len(self.coding_results['model_name'].unique())}\n"
            )
            f.write(f"Total reasoning samples: {len(self.coding_results)}\n\n")

            f.write("SELECTED TRANSCRIPTS\n")
            f.write("-" * 20 + "\n")
            for i, tid in enumerate(self.transcripts_selected, 1):
                f.write(f"  {i:2d}. {tid}\n")
            f.write("\n")

            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")

            # Rubric alignment findings
            f.write("1. RUBRIC ALIGNMENT QUALITY:\n")
            for model in self.coding_results["model_name"].unique():
                model_data = self.coding_results[
                    self.coding_results["model_name"] == model
                ]
                clear_pct = (model_data["rubric_alignment"] == "Clear").mean() * 100
                f.write(f"   {model}: {clear_pct:.1f}% clear alignment\n")
            f.write("\n")

            # Reasoning depth findings
            f.write("2. REASONING DEPTH:\n")
            for model in self.coding_results["model_name"].unique():
                model_data = self.coding_results[
                    self.coding_results["model_name"] == model
                ]
                indepth_pct = (model_data["reasoning_depth"] == "In-depth").mean() * 100
                f.write(f"   {model}: {indepth_pct:.1f}% in-depth reasoning\n")
            f.write("\n")

            # Length statistics
            f.write("3. RESPONSE LENGTH STATISTICS:\n")
            length_stats = self.coding_results.groupby("model_name")["word_count"].agg(
                ["mean", "std"]
            )
            for model, stats in length_stats.iterrows():
                f.write(f"   {model}: {stats['mean']:.1f} ± {stats['std']:.1f} words\n")
            f.write("\n")

            f.write("DETAILED STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(str(stats_dict["alignment_crosstab"].round(1)))
            f.write("\n\n")
            f.write(str(stats_dict["depth_crosstab"].round(1)))
            f.write("\n\n")

        print(f"✅ Comprehensive report saved to: {report_path}")
        return report_path


def main():
    """Main execution function."""
    print("🚀 Starting Experiment 4: Qualitative Reasoning Analysis")
    print("=" * 60)

    try:
        # Initialize analyzer
        analyzer = ReasoningAnalyzer()

        # Load data and select transcripts
        analyzer.load_data()
        analyzer.select_diverse_transcripts(n_transcripts=15)

        # Analyze reasoning patterns
        coding_results = analyzer.analyze_all_reasoning()

        # Statistical analysis
        stats_dict = analyzer.statistical_analysis()

        # Create visualizations
        analyzer.create_visualizations(stats_dict)

        # Generate qualitative examples
        analyzer.generate_qualitative_examples()

        # Save comprehensive report
        analyzer.save_comprehensive_report(stats_dict)

        print("\n" + "=" * 60)
        print("🎉 EXPERIMENT 4 COMPLETE! 🎉")
        print("=" * 60)
        print(
            f"📊 Analyzed reasoning patterns for {len(analyzer.transcripts_selected)} transcripts"
        )
        print(f"📈 Generated visualizations and comprehensive report")
        print(f"📁 Results saved to: {ANALYSIS_DIR}")

    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
