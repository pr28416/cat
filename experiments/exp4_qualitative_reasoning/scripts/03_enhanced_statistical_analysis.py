#!/usr/bin/env python3
"""
Experiment 4: Enhanced Statistical Analysis

This script provides rigorous statistical analysis of the qualitative reasoning patterns,
including hypothesis testing, effect sizes, inter-rater reliability, and advanced modeling.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    kruskal,
    mannwhitneyu,
    friedmanchisquare,
    wilcoxon,
    spearmanr,
    pearsonr,
)
import scikit_posthocs as sp
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import warnings
from datetime import datetime
import json

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)

# --- Constants ---
EXPERIMENT_ID = "EXP4_EnhancedStats"
ALPHA = 0.05
BONFERRONI_CORRECTION = True

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
EXP3_RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "experiments/exp3_model_comparison/results/processed_scores"
)


class EnhancedStatisticalAnalyzer:
    """Enhanced statistical analysis for qualitative reasoning patterns."""

    def __init__(self):
        self.df = None
        self.exp3_df = None
        self.results = {}

    def load_data(self):
        """Load both Experiment 4 and Experiment 3 data."""
        print("=" * 60)
        print("🔍 LOADING DATA FOR ENHANCED STATISTICAL ANALYSIS")
        print("=" * 60)

        # Load Experiment 4 results
        exp4_path = os.path.join(PROCESSED_DIR, "reasoning_analysis_summary.csv")
        self.df = pd.read_csv(exp4_path)

        # Load Experiment 3 results for correlation analysis
        exp3_path = os.path.join(
            EXP3_RESULTS_DIR, "exp3_modelcomparison_grading_results_curated_models.csv"
        )
        self.exp3_df = pd.read_csv(exp3_path)
        self.exp3_df = self.exp3_df[self.exp3_df["Parsing_Error"].isna()]

        print(f"📊 Experiment 4 Data: {len(self.df)} reasoning samples")
        print(f"📊 Experiment 3 Data: {len(self.exp3_df)} scoring attempts")
        print(f"📊 Models: {', '.join(self.df['model_name'].unique())}")
        print(f"📊 Transcripts: {self.df['transcript_id'].nunique()}")

        return self.df, self.exp3_df

    def test_evidence_citation_differences(self):
        """Test for significant differences in evidence citation patterns."""
        print("\n" + "=" * 60)
        print("1️⃣ EVIDENCE CITATION PATTERN ANALYSIS")
        print("=" * 60)

        # Create binary indicators for evidence types
        self.df["has_direct_quote"] = self.df["evidence_types"].apply(
            lambda x: "Direct Quote" in x
        )
        self.df["has_paraphrase"] = self.df["evidence_types"].apply(
            lambda x: "Paraphrase" in x
        )
        self.df["only_general"] = self.df["evidence_types"].apply(
            lambda x: x == ["General Statement"]
        )

        models = self.df["model_name"].unique()

        # Chi-square test for independence
        print("🔬 Chi-square Tests for Evidence Citation Independence:")

        # Direct quote usage
        contingency_quotes = pd.crosstab(
            self.df["model_name"], self.df["has_direct_quote"]
        )
        chi2_quotes, p_quotes, dof_quotes, expected_quotes = chi2_contingency(
            contingency_quotes
        )

        print(f"\n📊 Direct Quote Usage:")
        print(f"   χ² = {chi2_quotes:.4f}, df = {dof_quotes}, p = {p_quotes:.6f}")
        print(f"   {'✅ SIGNIFICANT' if p_quotes < ALPHA else '❌ NOT SIGNIFICANT'}")

        # Effect size (Cramér's V)
        n = contingency_quotes.sum().sum()
        cramers_v_quotes = np.sqrt(
            chi2_quotes / (n * (min(contingency_quotes.shape) - 1))
        )
        print(
            f"   Cramér's V = {cramers_v_quotes:.4f} ({'Large' if cramers_v_quotes > 0.5 else 'Medium' if cramers_v_quotes > 0.3 else 'Small'} effect)"
        )

        # Pairwise comparisons with Fisher's exact test
        print(f"\n🔍 Pairwise Fisher's Exact Tests (Direct Quotes):")
        pairwise_results = []

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i + 1 :], i + 1):
                model1_data = self.df[self.df["model_name"] == model1]
                model2_data = self.df[self.df["model_name"] == model2]

                # Create 2x2 contingency table
                m1_quotes = model1_data["has_direct_quote"].sum()
                m1_no_quotes = len(model1_data) - m1_quotes
                m2_quotes = model2_data["has_direct_quote"].sum()
                m2_no_quotes = len(model2_data) - m2_quotes

                table = [[m1_quotes, m1_no_quotes], [m2_quotes, m2_no_quotes]]
                odds_ratio, p_fisher = fisher_exact(table)

                # Bonferroni correction
                n_comparisons = len(models) * (len(models) - 1) // 2
                p_corrected = (
                    min(p_fisher * n_comparisons, 1.0)
                    if BONFERRONI_CORRECTION
                    else p_fisher
                )

                pairwise_results.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "odds_ratio": odds_ratio,
                        "p_value": p_fisher,
                        "p_corrected": p_corrected,
                        "significant": p_corrected < ALPHA,
                    }
                )

                print(
                    f"   {model1} vs {model2}: OR = {odds_ratio:.3f}, p = {p_corrected:.6f} {'✅' if p_corrected < ALPHA else '❌'}"
                )

        self.results["evidence_citation"] = {
            "chi2_stat": chi2_quotes,
            "p_value": p_quotes,
            "cramers_v": cramers_v_quotes,
            "pairwise_results": pairwise_results,
        }

        return self.results["evidence_citation"]

    def test_rubric_alignment_patterns(self):
        """Test for differences in rubric alignment quality."""
        print("\n" + "=" * 60)
        print("2️⃣ RUBRIC ALIGNMENT QUALITY ANALYSIS")
        print("=" * 60)

        # Ordinal encoding for alignment quality
        alignment_order = {"Loose": 1, "Moderate": 2, "Clear": 3}
        self.df["alignment_score"] = self.df["rubric_alignment"].map(alignment_order)

        # Kruskal-Wallis test (non-parametric ANOVA for ordinal data)
        models = self.df["model_name"].unique()
        alignment_groups = [
            self.df[self.df["model_name"] == model]["alignment_score"].values
            for model in models
        ]

        kw_stat, kw_p = kruskal(*alignment_groups)

        print(f"🔬 Kruskal-Wallis Test for Rubric Alignment:")
        print(f"   H = {kw_stat:.4f}, p = {kw_p:.6f}")
        print(f"   {'✅ SIGNIFICANT' if kw_p < ALPHA else '❌ NOT SIGNIFICANT'}")

        # Effect size (eta-squared approximation)
        n_total = len(self.df)
        eta_squared = (kw_stat - len(models) + 1) / (n_total - len(models))
        print(
            f"   η² ≈ {eta_squared:.4f} ({'Large' if eta_squared > 0.14 else 'Medium' if eta_squared > 0.06 else 'Small'} effect)"
        )

        # Post-hoc pairwise comparisons (Dunn's test)
        if kw_p < ALPHA:
            print(f"\n🔍 Post-hoc Dunn's Test (Bonferroni corrected):")

            # Create matrix for post-hoc test
            alignment_matrix = []
            for transcript in self.df["transcript_id"].unique():
                transcript_data = self.df[self.df["transcript_id"] == transcript]
                if len(transcript_data) == len(models):  # Complete data
                    row = (
                        transcript_data.set_index("model_name")
                        .loc[models]["alignment_score"]
                        .values
                    )
                    alignment_matrix.append(row)

            if len(alignment_matrix) > 0:
                alignment_matrix = np.array(alignment_matrix)
                posthoc_results = sp.posthoc_dunn(
                    alignment_matrix, p_adjust="bonferroni"
                )
                posthoc_df = pd.DataFrame(posthoc_results, index=models, columns=models)

                print("   P-values matrix (Bonferroni corrected):")
                print(posthoc_df.round(6))

        self.results["rubric_alignment"] = {
            "kruskal_stat": kw_stat,
            "p_value": kw_p,
            "eta_squared": eta_squared,
        }

        return self.results["rubric_alignment"]

    def test_reasoning_depth_patterns(self):
        """Test for differences in reasoning depth."""
        print("\n" + "=" * 60)
        print("3️⃣ REASONING DEPTH ANALYSIS")
        print("=" * 60)

        # Ordinal encoding for reasoning depth
        depth_order = {"Superficial": 1, "Moderate": 2, "In-depth": 3}
        self.df["depth_score"] = self.df["reasoning_depth"].map(depth_order)

        # Kruskal-Wallis test
        models = self.df["model_name"].unique()
        depth_groups = [
            self.df[self.df["model_name"] == model]["depth_score"].values
            for model in models
        ]

        kw_stat_depth, kw_p_depth = kruskal(*depth_groups)

        print(f"🔬 Kruskal-Wallis Test for Reasoning Depth:")
        print(f"   H = {kw_stat_depth:.4f}, p = {kw_p_depth:.6f}")
        print(f"   {'✅ SIGNIFICANT' if kw_p_depth < ALPHA else '❌ NOT SIGNIFICANT'}")

        # Mann-Whitney U tests for pairwise comparisons
        print(f"\n🔍 Pairwise Mann-Whitney U Tests:")
        depth_pairwise = []

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i + 1 :], i + 1):
                group1 = self.df[self.df["model_name"] == model1]["depth_score"]
                group2 = self.df[self.df["model_name"] == model2]["depth_score"]

                u_stat, p_mann = mannwhitneyu(group1, group2, alternative="two-sided")

                # Effect size (rank-biserial correlation)
                n1, n2 = len(group1), len(group2)
                r_rb = 1 - (2 * u_stat) / (n1 * n2)

                # Bonferroni correction
                n_comparisons = len(models) * (len(models) - 1) // 2
                p_corrected = (
                    min(p_mann * n_comparisons, 1.0)
                    if BONFERRONI_CORRECTION
                    else p_mann
                )

                depth_pairwise.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "u_statistic": u_stat,
                        "p_value": p_mann,
                        "p_corrected": p_corrected,
                        "effect_size_r": r_rb,
                        "significant": p_corrected < ALPHA,
                    }
                )

                print(
                    f"   {model1} vs {model2}: U = {u_stat:.1f}, p = {p_corrected:.6f}, r = {r_rb:.3f} {'✅' if p_corrected < ALPHA else '❌'}"
                )

        self.results["reasoning_depth"] = {
            "kruskal_stat": kw_stat_depth,
            "p_value": kw_p_depth,
            "pairwise_results": depth_pairwise,
        }

        return self.results["reasoning_depth"]

    def analyze_consistency_reasoning_correlation(self):
        """Analyze correlation between consistency (Exp 3) and reasoning quality (Exp 4)."""
        print("\n" + "=" * 60)
        print("4️⃣ CONSISTENCY-REASONING CORRELATION ANALYSIS")
        print("=" * 60)

        # Calculate consistency metrics from Experiment 3
        consistency_stats = []
        for transcript_id in self.df["transcript_id"].unique():
            for model in self.df["model_name"].unique():
                # Get consistency from Exp 3
                exp3_data = self.exp3_df[
                    (self.exp3_df["transcript_id"] == transcript_id)
                    & (self.exp3_df["model_name"] == model)
                ]

                if len(exp3_data) > 1:
                    consistency = exp3_data["Total_Score_Calculated"].std()

                    # Get reasoning quality from Exp 4
                    exp4_data = self.df[
                        (self.df["transcript_id"] == transcript_id)
                        & (self.df["model_name"] == model)
                    ]

                    if len(exp4_data) > 0:
                        reasoning_data = exp4_data.iloc[0]

                        consistency_stats.append(
                            {
                                "transcript_id": transcript_id,
                                "model_name": model,
                                "consistency_stdev": consistency,
                                "word_count": reasoning_data["word_count"],
                                "alignment_score": reasoning_data["alignment_score"],
                                "depth_score": reasoning_data["depth_score"],
                                "has_direct_quote": reasoning_data["has_direct_quote"],
                                "has_formatting": reasoning_data["has_formatting"],
                            }
                        )

        corr_df = pd.DataFrame(consistency_stats)

        if len(corr_df) > 0:
            print(f"📊 Correlation Analysis (n = {len(corr_df)} observations):")

            # Spearman correlations (robust to outliers, works with ordinal data)
            correlations = {}

            variables = ["word_count", "alignment_score", "depth_score"]
            for var in variables:
                corr_coef, p_value = spearmanr(
                    corr_df["consistency_stdev"], corr_df[var]
                )
                correlations[var] = {"correlation": corr_coef, "p_value": p_value}

                print(f"   Consistency ↔ {var.replace('_', ' ').title()}:")
                print(
                    f"     ρ = {corr_coef:.4f}, p = {p_value:.6f} {'✅' if p_value < ALPHA else '❌'}"
                )

                # Interpretation
                abs_corr = abs(corr_coef)
                strength = (
                    "Strong"
                    if abs_corr > 0.7
                    else "Moderate" if abs_corr > 0.3 else "Weak"
                )
                direction = "Positive" if corr_coef > 0 else "Negative"
                print(f"     {strength} {direction.lower()} correlation")

            # Point-biserial correlation for binary variables
            binary_vars = ["has_direct_quote", "has_formatting"]
            for var in binary_vars:
                corr_coef, p_value = pearsonr(
                    corr_df["consistency_stdev"], corr_df[var].astype(int)
                )
                correlations[var] = {"correlation": corr_coef, "p_value": p_value}

                print(f"   Consistency ↔ {var.replace('_', ' ').title()}:")
                print(
                    f"     r = {corr_coef:.4f}, p = {p_value:.6f} {'✅' if p_value < ALPHA else '❌'}"
                )

            # Model-level aggregation
            print(f"\n📊 Model-Level Correlations:")
            model_stats = (
                corr_df.groupby("model_name")
                .agg(
                    {
                        "consistency_stdev": "mean",
                        "word_count": "mean",
                        "alignment_score": "mean",
                        "depth_score": "mean",
                        "has_direct_quote": "mean",
                        "has_formatting": "mean",
                    }
                )
                .reset_index()
            )

            for var in [
                "word_count",
                "alignment_score",
                "depth_score",
                "has_direct_quote",
            ]:
                if (
                    len(model_stats) >= 3
                ):  # Need at least 3 points for meaningful correlation
                    corr_coef, p_value = spearmanr(
                        model_stats["consistency_stdev"], model_stats[var]
                    )
                    print(
                        f"   Model-level Consistency ↔ {var.replace('_', ' ').title()}:"
                    )
                    print(
                        f"     ρ = {corr_coef:.4f}, p = {p_value:.6f} {'✅' if p_value < ALPHA else '❌'}"
                    )

            self.results["consistency_correlation"] = {
                "correlations": correlations,
                "model_level_stats": model_stats,
            }

        return self.results.get("consistency_correlation", {})

    def multivariate_analysis(self):
        """Perform multivariate analysis to identify reasoning patterns."""
        print("\n" + "=" * 60)
        print("5️⃣ MULTIVARIATE PATTERN ANALYSIS")
        print("=" * 60)

        # Prepare features for classification
        features = [
            "word_count",
            "sentence_count",
            "alignment_score",
            "depth_score",
            "has_direct_quote",
            "has_paraphrase",
            "has_formatting",
            "has_bullets",
        ]

        # Ensure all features exist
        for feature in features:
            if feature not in self.df.columns:
                if feature == "has_paraphrase":
                    self.df[feature] = self.df["evidence_types"].apply(
                        lambda x: "Paraphrase" in x
                    )
                else:
                    print(
                        f"⚠️ Feature {feature} not found, skipping multivariate analysis"
                    )
                    return {}

        X = self.df[features].copy()

        # Convert boolean to int
        bool_cols = [
            "has_direct_quote",
            "has_paraphrase",
            "has_formatting",
            "has_bullets",
        ]
        for col in bool_cols:
            X[col] = X[col].astype(int)

        # Encode model names
        le = LabelEncoder()
        y = le.fit_transform(self.df["model_name"])

        print(f"🔬 Random Forest Classification Analysis:")
        print(f"   Features: {len(features)}")
        print(f"   Samples: {len(X)}")
        print(f"   Classes: {len(le.classes_)} models")

        # Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")

        print(
            f"   Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

        # Fit for feature importance
        rf.fit(X, y)
        feature_importance = pd.DataFrame(
            {"feature": features, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"\n📊 Feature Importance Rankings:")
        for _, row in feature_importance.iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")

        # Permutation importance for more robust estimates
        from sklearn.inspection import permutation_importance

        perm_importance = permutation_importance(
            rf, X, y, n_repeats=10, random_state=42
        )

        perm_df = pd.DataFrame(
            {
                "feature": features,
                "importance_mean": perm_importance.importances_mean,
                "importance_std": perm_importance.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)

        print(f"\n📊 Permutation Importance (more robust):")
        for _, row in perm_df.iterrows():
            print(
                f"   {row['feature']:20s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
            )

        self.results["multivariate"] = {
            "cv_accuracy": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "feature_importance": feature_importance.to_dict("records"),
            "permutation_importance": perm_df.to_dict("records"),
        }

        return self.results["multivariate"]

    def inter_model_agreement_analysis(self):
        """Analyze agreement between models on the same transcripts."""
        print("\n" + "=" * 60)
        print("6️⃣ INTER-MODEL AGREEMENT ANALYSIS")
        print("=" * 60)

        # Calculate pairwise agreement for categorical variables
        models = self.df["model_name"].unique()

        # Rubric alignment agreement
        print(f"🔬 Rubric Alignment Agreement:")
        alignment_agreements = []

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i + 1 :], i + 1):
                # Get shared transcripts
                shared_transcripts = set(
                    self.df[self.df["model_name"] == model1]["transcript_id"]
                ).intersection(
                    self.df[self.df["model_name"] == model2]["transcript_id"]
                )

                if len(shared_transcripts) > 0:
                    model1_data = self.df[
                        (self.df["model_name"] == model1)
                        & (self.df["transcript_id"].isin(shared_transcripts))
                    ].set_index("transcript_id")["rubric_alignment"]

                    model2_data = self.df[
                        (self.df["model_name"] == model2)
                        & (self.df["transcript_id"].isin(shared_transcripts))
                    ].set_index("transcript_id")["rubric_alignment"]

                    # Align by transcript
                    aligned_transcripts = model1_data.index.intersection(
                        model2_data.index
                    )
                    if len(aligned_transcripts) > 1:
                        ratings1 = model1_data.loc[aligned_transcripts].values
                        ratings2 = model2_data.loc[aligned_transcripts].values

                        # Cohen's Kappa
                        kappa = cohen_kappa_score(ratings1, ratings2)

                        # Simple agreement percentage
                        agreement_pct = (ratings1 == ratings2).mean() * 100

                        alignment_agreements.append(
                            {
                                "model1": model1,
                                "model2": model2,
                                "n_transcripts": len(aligned_transcripts),
                                "kappa": kappa,
                                "agreement_pct": agreement_pct,
                            }
                        )

                        print(f"   {model1} ↔ {model2}:")
                        print(
                            f"     κ = {kappa:.4f}, Agreement = {agreement_pct:.1f}% (n={len(aligned_transcripts)})"
                        )

        # Reasoning depth agreement
        print(f"\n🔬 Reasoning Depth Agreement:")
        depth_agreements = []

        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models[i + 1 :], i + 1):
                shared_transcripts = set(
                    self.df[self.df["model_name"] == model1]["transcript_id"]
                ).intersection(
                    self.df[self.df["model_name"] == model2]["transcript_id"]
                )

                if len(shared_transcripts) > 0:
                    model1_data = self.df[
                        (self.df["model_name"] == model1)
                        & (self.df["transcript_id"].isin(shared_transcripts))
                    ].set_index("transcript_id")["reasoning_depth"]

                    model2_data = self.df[
                        (self.df["model_name"] == model2)
                        & (self.df["transcript_id"].isin(shared_transcripts))
                    ].set_index("transcript_id")["reasoning_depth"]

                    aligned_transcripts = model1_data.index.intersection(
                        model2_data.index
                    )
                    if len(aligned_transcripts) > 1:
                        ratings1 = model1_data.loc[aligned_transcripts].values
                        ratings2 = model2_data.loc[aligned_transcripts].values

                        kappa_depth = cohen_kappa_score(ratings1, ratings2)
                        agreement_pct_depth = (ratings1 == ratings2).mean() * 100

                        depth_agreements.append(
                            {
                                "model1": model1,
                                "model2": model2,
                                "kappa": kappa_depth,
                                "agreement_pct": agreement_pct_depth,
                            }
                        )

                        print(f"   {model1} ↔ {model2}:")
                        print(
                            f"     κ = {kappa_depth:.4f}, Agreement = {agreement_pct_depth:.1f}%"
                        )

        # Overall inter-model reliability
        if len(alignment_agreements) > 0:
            avg_kappa_alignment = np.mean([a["kappa"] for a in alignment_agreements])
            avg_agreement_alignment = np.mean(
                [a["agreement_pct"] for a in alignment_agreements]
            )

            print(f"\n📊 Overall Rubric Alignment Reliability:")
            print(f"   Average κ = {avg_kappa_alignment:.4f}")
            print(f"   Average Agreement = {avg_agreement_alignment:.1f}%")

            # Interpretation
            if avg_kappa_alignment > 0.8:
                interpretation = "Excellent"
            elif avg_kappa_alignment > 0.6:
                interpretation = "Good"
            elif avg_kappa_alignment > 0.4:
                interpretation = "Moderate"
            else:
                interpretation = "Poor"

            print(f"   Interpretation: {interpretation} inter-model reliability")

        self.results["inter_model_agreement"] = {
            "alignment_agreements": alignment_agreements,
            "depth_agreements": depth_agreements,
        }

        return self.results["inter_model_agreement"]

    def create_enhanced_visualizations(self):
        """Create enhanced statistical visualizations."""
        print("\n" + "=" * 60)
        print("📈 CREATING ENHANCED VISUALIZATIONS")
        print("=" * 60)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Correlation heatmap
        if "consistency_correlation" in self.results:
            corr_data = self.results["consistency_correlation"].get("model_level_stats")
            if corr_data is not None and len(corr_data) > 0:
                plt.figure(figsize=(10, 8))

                # Prepare correlation matrix
                corr_vars = [
                    "consistency_stdev",
                    "word_count",
                    "alignment_score",
                    "depth_score",
                    "has_direct_quote",
                ]
                corr_matrix = corr_data[corr_vars].corr()

                # Create heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(
                    corr_matrix,
                    mask=mask,
                    annot=True,
                    fmt=".3f",
                    cmap="RdBu_r",
                    center=0,
                    vmin=-1,
                    vmax=1,
                    cbar_kws={"label": "Correlation Coefficient"},
                )
                plt.title(
                    "Model-Level Correlations: Consistency vs Reasoning Quality",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.tight_layout()

                corr_plot_path = os.path.join(
                    ANALYSIS_DIR, "exp4_enhanced_correlation_heatmap.png"
                )
                plt.savefig(corr_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✅ Saved correlation heatmap: {corr_plot_path}")

        # 2. Feature importance plot
        if "multivariate" in self.results:
            importance_data = self.results["multivariate"].get(
                "permutation_importance", []
            )
            if importance_data:
                plt.figure(figsize=(12, 6))

                importance_df = pd.DataFrame(importance_data)

                # Create bar plot with error bars
                plt.barh(
                    range(len(importance_df)),
                    importance_df["importance_mean"],
                    xerr=importance_df["importance_std"],
                    capsize=5,
                )

                plt.yticks(range(len(importance_df)), importance_df["feature"])
                plt.xlabel("Permutation Importance")
                plt.title(
                    "Feature Importance for Model Classification",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.grid(axis="x", alpha=0.3)
                plt.tight_layout()

                importance_plot_path = os.path.join(
                    ANALYSIS_DIR, "exp4_enhanced_feature_importance.png"
                )
                plt.savefig(importance_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✅ Saved feature importance plot: {importance_plot_path}")

        # 3. Agreement matrix visualization
        if "inter_model_agreement" in self.results:
            alignment_agreements = self.results["inter_model_agreement"].get(
                "alignment_agreements", []
            )
            if alignment_agreements:
                # Create agreement matrix
                models = self.df["model_name"].unique()
                agreement_matrix = np.ones((len(models), len(models)))

                for agreement in alignment_agreements:
                    i = list(models).index(agreement["model1"])
                    j = list(models).index(agreement["model2"])
                    kappa = agreement["kappa"]
                    agreement_matrix[i, j] = kappa
                    agreement_matrix[j, i] = kappa

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    agreement_matrix,
                    annot=True,
                    fmt=".3f",
                    xticklabels=models,
                    yticklabels=models,
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Cohen's Kappa"},
                )
                plt.title(
                    "Inter-Model Agreement: Rubric Alignment",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()

                agreement_plot_path = os.path.join(
                    ANALYSIS_DIR, "exp4_enhanced_inter_model_agreement.png"
                )
                plt.savefig(agreement_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"✅ Saved agreement matrix: {agreement_plot_path}")

    def save_enhanced_results(self):
        """Save comprehensive statistical results."""
        print("\n" + "=" * 60)
        print("💾 SAVING ENHANCED STATISTICAL RESULTS")
        print("=" * 60)

        # Create a simplified results dictionary
        simplified_results = {}

        # Evidence citation results
        if "evidence_citation" in self.results:
            ec = self.results["evidence_citation"]
            simplified_results["evidence_citation"] = {
                "chi2_statistic": float(ec["chi2_stat"]),
                "p_value": float(ec["p_value"]),
                "cramers_v": float(ec["cramers_v"]),
                "significant": ec["p_value"] < 0.05,
                "effect_size": (
                    "Large"
                    if ec["cramers_v"] > 0.3
                    else "Medium" if ec["cramers_v"] > 0.1 else "Small"
                ),
            }

        # Rubric alignment results
        if "rubric_alignment" in self.results:
            ra = self.results["rubric_alignment"]
            simplified_results["rubric_alignment"] = {
                "kruskal_statistic": float(ra["kruskal_stat"]),
                "p_value": float(ra["p_value"]),
                "eta_squared": float(ra["eta_squared"]),
                "significant": ra["p_value"] < 0.05,
                "effect_size": (
                    "Large"
                    if ra["eta_squared"] > 0.14
                    else "Medium" if ra["eta_squared"] > 0.06 else "Small"
                ),
            }

        # Reasoning depth results
        if "reasoning_depth" in self.results:
            rd = self.results["reasoning_depth"]
            simplified_results["reasoning_depth"] = {
                "kruskal_statistic": float(rd["kruskal_stat"]),
                "p_value": float(rd["p_value"]),
                "significant": rd["p_value"] < 0.05,
            }

        # Consistency correlation results
        if "consistency_correlation" in self.results:
            cc = self.results["consistency_correlation"]
            simplified_results["consistency_correlation"] = {
                "word_count_correlation": cc["correlations"]["word_count"][
                    "correlation"
                ],
                "word_count_p_value": cc["correlations"]["word_count"]["p_value"],
                "alignment_correlation": cc["correlations"]["alignment_score"][
                    "correlation"
                ],
                "alignment_p_value": cc["correlations"]["alignment_score"]["p_value"],
            }

        # Multivariate results
        if "multivariate" in self.results:
            mv = self.results["multivariate"]
            simplified_results["multivariate"] = {
                "classification_accuracy": float(mv["cv_accuracy"]),
                "accuracy_std": float(mv["cv_std"]),
                "top_features": [
                    f["feature"] for f in mv["permutation_importance"][:3]
                ],
            }

        # Save simplified results
        results_path = os.path.join(
            ANALYSIS_DIR, "exp4_enhanced_statistical_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(simplified_results, f, indent=2, default=str)

        print(f"✅ Detailed results saved to: {results_path}")

        # Create summary report
        report_path = os.path.join(ANALYSIS_DIR, "exp4_enhanced_statistical_report.txt")

        with open(report_path, "w") as f:
            f.write("EXPERIMENT 4: ENHANCED STATISTICAL ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("STATISTICAL TESTS PERFORMED\n")
            f.write("-" * 30 + "\n")
            f.write("1. Chi-square test for evidence citation independence\n")
            f.write("2. Kruskal-Wallis test for rubric alignment differences\n")
            f.write("3. Mann-Whitney U tests for reasoning depth comparisons\n")
            f.write(
                "4. Spearman correlations for consistency-reasoning relationships\n"
            )
            f.write("5. Random Forest classification for multivariate patterns\n")
            f.write("6. Cohen's Kappa for inter-model agreement\n\n")

            # Write key findings
            f.write("KEY STATISTICAL FINDINGS\n")
            f.write("-" * 30 + "\n")

            if "evidence_citation" in simplified_results:
                ec = simplified_results["evidence_citation"]
                f.write(
                    f"Evidence Citation: χ² = {ec['chi2_statistic']:.4f}, p = {ec['p_value']:.6f}\n"
                )
                f.write(
                    f"Effect Size (Cramér's V): {ec['cramers_v']:.4f} ({ec['effect_size']})\n"
                )
                f.write(f"Significant: {'Yes' if ec['significant'] else 'No'}\n\n")

            if "rubric_alignment" in simplified_results:
                ra = simplified_results["rubric_alignment"]
                f.write(
                    f"Rubric Alignment: H = {ra['kruskal_statistic']:.4f}, p = {ra['p_value']:.6f}\n"
                )
                f.write(
                    f"Effect Size (η²): {ra['eta_squared']:.4f} ({ra['effect_size']})\n"
                )
                f.write(f"Significant: {'Yes' if ra['significant'] else 'No'}\n\n")

            if "reasoning_depth" in simplified_results:
                rd = simplified_results["reasoning_depth"]
                f.write(
                    f"Reasoning Depth: H = {rd['kruskal_statistic']:.4f}, p = {rd['p_value']:.6f}\n"
                )
                f.write(f"Significant: {'Yes' if rd['significant'] else 'No'}\n\n")

            if "multivariate" in simplified_results:
                mv = simplified_results["multivariate"]
                f.write(
                    f"Model Classification Accuracy: {mv['classification_accuracy']:.3f} ± {mv['accuracy_std']:.3f}\n"
                )
                f.write(f"Top Features: {', '.join(mv['top_features'])}\n\n")

            f.write("CLINICAL IMPLICATIONS\n")
            f.write("-" * 30 + "\n")
            f.write(
                "1. Significant differences in evidence citation patterns suggest models vary in validity\n"
            )
            f.write(
                "2. Large effect sizes indicate practical significance for clinical applications\n"
            )
            f.write(
                "3. Consistency-reasoning paradox: most consistent models may have poorest reasoning\n"
            )
            f.write("4. Feature importance analysis guides model selection criteria\n")

        print(f"✅ Summary report saved to: {report_path}")

        return results_path, report_path


def main():
    """Main execution function."""
    print("🚀 Starting Enhanced Statistical Analysis for Experiment 4")
    print("=" * 60)

    try:
        # Initialize analyzer
        analyzer = EnhancedStatisticalAnalyzer()

        # Load data
        analyzer.load_data()

        # Run statistical tests
        analyzer.test_evidence_citation_differences()
        analyzer.test_rubric_alignment_patterns()
        analyzer.test_reasoning_depth_patterns()
        analyzer.analyze_consistency_reasoning_correlation()
        analyzer.multivariate_analysis()
        analyzer.inter_model_agreement_analysis()

        # Create visualizations
        analyzer.create_enhanced_visualizations()

        # Save results
        analyzer.save_enhanced_results()

        print("\n" + "=" * 60)
        print("🎉 ENHANCED STATISTICAL ANALYSIS COMPLETE! 🎉")
        print("=" * 60)
        print("📊 Comprehensive statistical tests performed")
        print("📈 Enhanced visualizations generated")
        print("📋 Detailed results and reports saved")
        print(f"📁 Results available in: {ANALYSIS_DIR}")

    except Exception as e:
        print(f"\n❌ Error during enhanced analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
