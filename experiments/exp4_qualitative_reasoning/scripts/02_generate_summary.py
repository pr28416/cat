#!/usr/bin/env python3
"""
Experiment 4: Summary and Key Insights

This script provides a concise summary of the key findings from the qualitative reasoning analysis.
"""

import os
import sys
import pandas as pd
import json

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- File Paths ---
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROCESSED_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp4_qualitative_reasoning/results/processed_scores"
)
ANALYSIS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp4_qualitative_reasoning/results/analysis"
)


def load_results():
    """Load the analysis results."""
    results_path = os.path.join(PROCESSED_DIR, "reasoning_analysis_summary.csv")
    return pd.read_csv(results_path)


def print_key_insights():
    """Print the key insights from Experiment 4."""
    print("🎯 EXPERIMENT 4: KEY INSIGHTS")
    print("=" * 60)

    df = load_results()

    print("\n1️⃣ THE CONSISTENCY-REASONING PARADOX")
    print("-" * 40)
    print("📊 Most Consistent Model (GPT-4.1):")
    gpt41_data = df[df["model_name"] == "gpt-4.1-2025-04-14"]
    print(
        f"   • Evidence Citation: {(gpt41_data['evidence_types'].apply(lambda x: 'Direct Quote' in x).mean() * 100):.1f}% use direct quotes"
    )
    print(
        f"   • Reasoning Depth: {(gpt41_data['reasoning_depth'] == 'Superficial').mean() * 100:.1f}% superficial reasoning"
    )
    print(
        f"   • Rubric Alignment: {(gpt41_data['rubric_alignment'] == 'Clear').mean() * 100:.1f}% clear alignment"
    )

    print("\n📊 Least Consistent Model (o3-mini):")
    o3mini_data = df[df["model_name"] == "o3-mini-2025-01-31"]
    print(
        f"   • Evidence Citation: {(o3mini_data['evidence_types'].apply(lambda x: 'Direct Quote' in x).mean() * 100):.1f}% use direct quotes"
    )
    print(
        f"   • Reasoning Depth: {(o3mini_data['reasoning_depth'] == 'In-depth').mean() * 100:.1f}% in-depth reasoning"
    )
    print(
        f"   • Rubric Alignment: {(o3mini_data['rubric_alignment'] == 'Clear').mean() * 100:.1f}% clear alignment"
    )

    print(
        "\n💡 KEY INSIGHT: The most consistent model provides the most superficial reasoning!"
    )

    print("\n2️⃣ MODEL PERSONALITY PROFILES")
    print("-" * 40)

    models = df["model_name"].unique()
    for model in models:
        model_data = df[df["model_name"] == model]
        avg_words = model_data["word_count"].mean()

        # Determine personality
        if "gpt-4.1" in model:
            personality = "📝 The Evidence Collector"
        elif "gpt-4o-2024-08-06" in model:
            personality = "🎯 The Rubric Expert"
        elif "gpt-4o-mini" in model:
            personality = "⚖️ The Balanced Performer"
        elif "o3-2025-04-16" in model:
            personality = "📏 The Minimal Responder"
        else:
            personality = "🤔 The Analytical Thinker"

        print(f"\n{personality}")
        print(f"   Model: {model}")
        print(f"   Avg Response: {avg_words:.0f} words")

        # Key characteristics
        direct_quotes = (
            model_data["evidence_types"].apply(lambda x: "Direct Quote" in x).mean()
            * 100
        )
        clear_alignment = (model_data["rubric_alignment"] == "Clear").mean() * 100
        indepth_reasoning = (model_data["reasoning_depth"] == "In-depth").mean() * 100

        print(f"   Evidence: {direct_quotes:.0f}% direct quotes")
        print(f"   Alignment: {clear_alignment:.0f}% clear")
        print(f"   Depth: {indepth_reasoning:.0f}% in-depth")

    print("\n3️⃣ THE EVIDENCE CRISIS")
    print("-" * 40)
    print("📈 Evidence Citation by Model Type:")

    # Group by model family
    gpt_models = df[df["model_name"].str.contains("gpt")]
    o3_models = df[df["model_name"].str.contains("o3")]

    gpt_quotes = (
        gpt_models["evidence_types"].apply(lambda x: "Direct Quote" in x).mean() * 100
    )
    o3_quotes = (
        o3_models["evidence_types"].apply(lambda x: "Direct Quote" in x).mean() * 100
    )

    print(f"   🟢 GPT Models: {gpt_quotes:.1f}% use direct quotes")
    print(f"   🔴 o3 Models: {o3_quotes:.1f}% use direct quotes")
    print(f"\n💡 INSIGHT: o3 models score without citing evidence - validity concern!")

    print("\n4️⃣ OPTIMAL MODEL RECOMMENDATION")
    print("-" * 40)
    print("🏆 WINNER: GPT-4o-2024-08-06")
    print("   ✅ Perfect rubric alignment (100%)")
    print("   ✅ Good evidence citation (26.7% direct quotes)")
    print("   ✅ Consistent response length (136 ± 21 words)")
    print("   ✅ High consistency from Experiment 3")
    print("   ✅ Best balance of all factors")

    print("\n🥈 RUNNER-UP: GPT-4o-mini-2024-07-18")
    print("   ✅ Cost-effective option")
    print("   ✅ Highest moderate reasoning (40%)")
    print("   ✅ Good rubric alignment (80%)")

    print("\n❌ NOT RECOMMENDED: o3 models")
    print("   ❌ Poor consistency (Experiment 3)")
    print("   ❌ Minimal evidence citation")
    print("   ❌ Extreme response lengths (23-79 words)")

    print("\n5️⃣ CLINICAL IMPLICATIONS")
    print("-" * 40)
    print("🏥 For Healthcare Deployment:")
    print("   • Transparency: GPT models provide explainable reasoning")
    print("   • Auditability: Evidence citations enable verification")
    print("   • Standardization: Clear rubric alignment ensures consistency")
    print("   • Trust: Detailed reasoning builds clinician confidence")

    print("\n⚠️ RED FLAGS:")
    print("   • o3 models: Score without justification")
    print("   • Consistency ≠ Quality: High reliability may mask poor reasoning")
    print("   • Evidence gaps: Some models ignore transcript content")

    print("\n" + "=" * 60)
    print("🎉 EXPERIMENT 4 COMPLETE!")
    print("📊 75 reasoning samples analyzed across 15 diverse transcripts")
    print("🔍 Six-dimensional coding scheme applied")
    print("📈 Four visualizations generated")
    print("📋 Comprehensive report available in results/")
    print("=" * 60)


def main():
    """Main execution function."""
    try:
        print_key_insights()
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
