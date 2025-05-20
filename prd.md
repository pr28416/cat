## **Product Requirements Document: LLM-Driven Health Communication Assessment Framework**

**1\. Introduction & Goals**

- **Project Title:** Optimizing a Large Language Model (LLM)-Driven Framework for Quantifiable Assessment of Patient Health Communication Proficiency.
- **Problem Statement:** Effective patient health communication is crucial for optimal outcomes, yet current assessment methods are often subjective, resource-intensive, and lack scalability and objective quantification. This project aims to address these limitations by developing a robust framework using LLMs as evaluative judges.
- **Project Goals:**
  1. To systematically develop and validate a methodology for employing Large Language Models as evaluative judges (LLM-as-a-Judge) to assess patient health communication proficiency based on the standardized "Patient Health Communication Rubric v5.0".
  2. To identify optimal configurations of LLM architecture, prompt engineering strategies, and LLM settings (e.g., temperature, max tokens) for maximizing scoring consistency, reliability, and exploring aspects of fairness and explainability in rubric application.
  3. To produce a robust, optimized LLM-driven assessment framework and tool, benchmarking its performance characteristics, including reliability, rubric adherence on synthetic data, qualitative reasoning, and potential for confidence estimation.
- **Target Outcomes:**
  1. A finalized, validated methodology for LLM-as-a-Judge based patient communication assessment using "Patient Health Communication Rubric v5.0".
  2. A framework and empirically supported guidelines for selecting effective LLM models and prompt engineering techniques for this task, recognizing the evolving nature of LLM capabilities.
  3. Quantitative benchmarks for the consistency (e.g., Standard Deviation of scores), rubric adherence (via synthetic data performance), qualitative reasoning alignment, and, where feasible, initial characterization of uncertainty/confidence of the optimized tool.
  4. A comprehensive dataset of experimental results (including synthetic transcripts and anonymized scoring logs) suitable for academic publication and supporting independent replication.
  5. A draft manuscript based on the findings, suitable for submission to a peer-reviewed journal (e.g., a high-impact medical informatics or clinical communication journal), acknowledging the scope of validation performed.
- **Success Metrics:**
  1. Statistically significant and meaningful differences identified between experimental conditions (e.g., prompt strategies, LLM models) in terms of scoring consistency and rubric adherence.
  2. Achieving a mean per-category score Standard Deviation (STDEV) ≤ 0.40 and a total-score STDEV ≤ 1.0 (or other pre-defined, justified thresholds) across at least 95% of transcripts in the held-out validation set (Experiment 5), using the optimized tool configuration.
  3. Demonstrated rubric adherence through a high correlation (e.g., Pearson r ≥ 0.7) between LLM-assigned total scores and target total scores on the synthetic transcript dataset (Experiment 1).
  4. Qualitative evidence from research team review of LLM reasoning (Experiment 4) demonstrating appropriate application of rubric criteria and evidence citation for a diverse subset of at least 15 real transcripts.
  5. Successful exploration and reporting of at least one feasible method for quantifying LLM score uncertainty/confidence on the validation set (Experiment 5), with an initial characterization of these properties (e.g., using analysis of output probabilities if available, or consistency across multiple runs).
  6. If demographic metadata is available and ethically usable for such analysis: No statistically significant detrimental difference (α = 0.05) in total-score distributions or key performance metrics across major patient demographic strata available in the dataset.
  7. Completion of all planned experiments and generation of all specified deliverables, including the open-source release of the core assessment pipeline components where feasible.

**2\. Target Audience (for the research and resulting tool)**

- Health literacy researchers
- Medical educators and curriculum developers
- Healthcare institutions and quality improvement programs
- Developers of digital health interventions

**3\. Core Component: Rubric for Patient Health Communication Assessment**

- **Name:** "Patient Health Communication Rubric v5.0"
- **Development Philosophy:** While "Rubric v5.0" is considered final for the experiments in this PRD, its initial development (and any future iterations beyond this project) should be grounded in established health literacy theories, communication analysis best practices, and an iterative process involving multidisciplinary expert review (clinicians, health literacy researchers, linguists) and empirical testing. Methodologies like those used for the Lasater Clinical Judgment Rubric (LCJR) or the REFLECT rubric, involving cycles of deductive/abductive analysis, empirical validation, and theory-informed design, serve as valuable models (Lit Review: Sources 35, 36).
- **Full Rubric Details:**
  - **Overall Structure:** 5 categories, each scored 1 (Poor) to 4 (Excellent). Total score range: 5-20.
  - **Categories & Scoring Criteria:**
    - **Criterion 1: Clarity of Language**
      - 1 (Poor): Frequently unclear or provides irrelevant information; responses are difficult to understand or follow. _Example: When asked about dietary habits, responds with, "I like food, but sometimes I don't eat it," without clarifying frequency or types of meals._
      - 2 (Fair): Occasionally unclear or includes minor irrelevant information; some evidence of misunderstanding. _Example: Describes dietary habits as "I eat healthy stuff like green things most days," without specifying foods or portions._
      - 3 (Good): Generally clear and relevant; minor ambiguities that do not significantly hinder understanding. _Example: States, "I usually eat vegetables and lean proteins, but I occasionally have snacks."_
      - 4 (Excellent): Consistently clear, concise, and focused; responses are entirely relevant and easy to understand. _Example: Explains, "I follow a balanced diet consisting of fruits, vegetables, lean proteins, and whole grains daily, with an occasional treat once or twice a week."_
    - **Criterion 2: Lexical Diversity**
      - 1 (Poor): Very limited vocabulary; repetitive or overly simplistic; lacks ability to express nuanced ideas. _Example: Repeatedly uses "good" to describe feelings and symptoms without elaboration._
      - 2 (Fair): Basic vocabulary with frequent repetition; struggles to convey complex or specific information. _Example: States, "I feel tired sometimes and other times not," without specifying frequency or context._
      - 3 (Good): Moderate vocabulary; some variation; appropriate for context but may lack precise terminology. _Example: Describes fatigue as "I feel tired after work, especially if I didn't sleep well the night before."_
      - 4 (Excellent): Rich and varied vocabulary; uses precise and contextually appropriate terms to effectively communicate. _Example: Explains, "I experience chronic fatigue, particularly in the evenings, which I attribute to insufficient rest and prolonged screen exposure."_
    - **Criterion 3: Conciseness and Completeness**
      - 1 (Poor): Response lacks critical information, leaving the core issue unclear or requiring significant prompting to clarify. _Example: "My stomach hurts." Does not provide additional context, such as duration, triggers, or associated symptoms._
      - 2 (Fair): Response includes some relevant details but misses key components or is vague, requiring follow-up questions. _Example: "I get stomach pain after eating spicy food." Does not address frequency, severity, or mitigating factors._
      - 3 (Good): Provides relevant information with minor omissions or occasional inclusion of extraneous details, but overall meaning is clear. _Example: "I often feel stomach pain after eating spicy food, lasting a couple of hours." May not specify severity or exact triggers._
      - 4 (Excellent): Response is comprehensive, including all critical details, with no unnecessary elaboration. _Example: "I experience sharp stomach pain after eating spicy food, especially chili peppers, usually within 30 minutes. The pain lasts 2-3 hours and improves with antacids."_
    - **Criterion 4: Engagement with Health Information**
      - 1 (Poor): Responses are brief, lack detail, and show no active effort to engage with the topic. Appears disinterested or uninvested. _Example: Provides one-word answers like, "Yes" or "No," when asked about symptoms, or says, "I don't know."_
      - 2 (Fair): Responds adequately to questions but rarely provides additional details or context. _Example: When asked about exercise habits, says, "I walk sometimes." Does not elaborate on frequency or duration._
      - 3 (Good): Provides relevant and reasonably detailed answers, demonstrating interest and understanding but rarely initiates clarifications or adds supplementary information. _Example: Says, "I walk about 3-4 times a week for 30 minutes." Responds well to questions but does not ask any._
      - 4 (Excellent): Offers detailed, thoughtful responses, initiates clarification or supplementary information, and asks relevant questions to deepen understanding. _Example: Explains, "I walk 3-4 times a week for about 30 minutes, usually after work. Should I increase the duration or add strength training for better results?"_
    - **Criterion 5: Health Literacy Indicator**
      - 1 (Poor): Lacks understanding of basic health concepts, frequently misinterprets questions. _Example: Misunderstands the term "cholesterol" as exclusively related to weight rather than blood lipid levels._
      - 2 (Fair): Limited understanding, may require simplification of health terms or further questioning to clarify answers. _Example: Says, "I think cholesterol is bad, but I'm not sure what foods have it."_
      - 3 (Good): Basic understanding, uses relevant health-related terms and describes symptoms appropriately. _Example: States, "I'm trying to avoid high-cholesterol foods like fried items and red meat."_
      - 4 (Excellent): Strong understanding of medical terminology and health context, with clear articulation of symptoms and health history. _Example: Explains, "I limit my intake of foods high in LDL cholesterol, such as processed meats and fried foods, and focus on incorporating HDL-boosting items like nuts and avocados."_
- **Statement of Finality:** This "Patient Health Communication Rubric v5.0" is considered final and will remain constant for all experiments detailed in this PRD to ensure comparability of results.

**4\. Core Component: Transcript Datasets**

- **Source 1 (DoVA):**
  - Description: De-identified patient-physician interaction transcripts from the US Department of Veterans Affairs.
  - Quantity: N \= 405
  - Access: Assumed to be publicly available or accessible via a specific provided path/API by the executing AI.
- **Source 2 (Nature Paper):**
  - Description: De-identified patient-physician interaction transcripts from \[User to provide specific Nature Paper Citation, e.g., "Smith et al., Nature Medicine, 20XX"\].
  - Quantity: N \= 272
  - Access: Assumed accessible via a specific provided path/API, with necessary permissions confirmed.
- **Data Acquisition Note:** The development of robust LLM-based health assessment tools is often constrained by the scarcity of suitable, large-scale, publicly available, and appropriately annotated datasets of real-world clinical conversations. This project leverages existing specified datasets; however, broader progress in the field may depend on collaborative efforts to create such resources (Lit Review: Sec 7.1, Insight 7.1).
- **Total Real Transcripts:** N_total_real \= 405 \+ 272 \= 677
- **De-identification & Ethical Considerations:** All transcripts are confirmed to be de-identified. Ethical approval or waiver for the use of these de-identified datasets for research purposes is documented (e.g., IRB Ref: \[User to provide if applicable\]).
  - **De-identification Challenges:** The process of de-identifying health data to comply with regulations like HIPAA and GDPR is critical but presents challenges. These include the risk of re-identification, the administrative burden, and the tension between rigorous privacy protection and retaining the linguistic richness necessary for nuanced LLM analysis. Over-aggressive de-identification might remove subtle cues vital for health literacy judgment (Lit Review: Sec 4.1.1, Sec 6.1, Sources 30, 31). This project relies on prior de-identification of the datasets.
- **Transcript Pre-processing:**
  - Ensure all transcripts are in a standardized plain text format (UTF-8 encoding).
  - Remove any explicit file headers or footers not part of the interaction.
  - Each transcript should have a unique persistent identifier (TranscriptID).
- **Data Partitioning Plan:**
  - Initialize a master list of all 677 TranscriptIDs.
  - **Exp 2 Set (Set A):** Randomly select 50 TranscriptIDs from the master list without replacement.
  - **Exp 3 Set (Set B):** Randomly select 50 TranscriptIDs from the _remaining_ master list without replacement (ensure no overlap with Set A).
  - **Exp 5 Set (Set C):** All remaining TranscriptIDs (N \= 677 \- 50 \- 50 \= 577\) will constitute Set C.
  - The randomization seed for selection must be recorded: SEED_DATA_PARTITION \= 12345\.

**5\. Core Component: LLM Models & Access**

- **List of Models (Available for Experiments):**
  - gpt-4o-2024-08-06
  - o4-mini-2025-04-16
  - o3-2025-04-16
  - gpt-4.1-mini-2025-04-14
  - _Contingency:_ If a specific model is unavailable, log the error and consult contingency plan (Section 12). The experimental plan (Exp3) aims to compare available models rather than depending on one specific architecture, providing inherent flexibility.
  - _Modeling Strategy Note:_ This project will primarily utilize prompt engineering with these general-purpose LLMs. While fine-tuning specialized LLM judges can offer benefits (Lit Review: Sec 2.2, Sources 15, 17), it is beyond the scope of the current experimental plan due to data and resource requirements. The developed methodology, however, could be extended to incorporate fine-tuned models in future work.
- **API Access:** API keys and endpoints for OpenAI (and other providers for any non-OpenAI models if they were to be included later, though current list is OpenAI-centric) are assumed to be securely provided to the executing AI.
- **Default Parameters (to be used unless an experiment manipulates them):**
  - **Temperature:** 0.1 (for maximizing reproducibility and minimizing randomness)
  - **Max Output Tokens:** Sufficient for scores and reasoning (e.g., 1000 tokens).
  - Other parameters (top_p, etc.): Use API defaults unless specified.
- **Uncertainty/Confidence Exploration:** While not a default parameter for all initial runs, Experiment 5 will specifically explore methods to quantify the uncertainty or confidence of the selected LLM's scores, if model outputs (e.g., token probabilities, alternative decodings) or API features allow for such analysis (Lit Review: Sec 5.2).
- All API calls must be logged with request, response, timestamp, model used, and cost (if retrievable).

**6\. Experimental Plan**

7.1. Experiment 1: Baseline LLM-Rubric Utility & Synthetic Data Efficacy  
\* 7.1.1. Experiment ID & Title: EXP1_BaselineRubricUtility  
\* 7.1.2. Objective(s):  
1\. Establish if the LLM utilizes Rubric 5.0 meaningfully for scoring (i.e., with improved consistency and alignment to rubric-defined targets) compared to an unguided LLM assessment.  
2\. Assess the LLM's ability to accurately and consistently score synthetic transcripts (generated with rubric-derived target scores) when explicitly using Rubric 5.0, thereby validating its adherence to the rubric.  
\* 7.1.3. Hypothesis(es):  
1\. H1a: Rubric-based grading (G2) will yield significantly lower Standard Deviation (STDEV) of total scores for the same transcript compared to non-rubric-based grading (G1).  
2\. H1b: For synthetic transcripts, rubric-based LLM total scores (G2) will exhibit a significantly lower Mean Absolute Error (MAE) from the synthetic (rubric-derived) target score compared to non-rubric-based scores (G1), indicating more accurate application of the rubric's principles.  
3\. H1c: Rubric-based grading (G2) will result in a distribution of total scores that is more aligned with the distribution of synthetic target scores than non-rubric-based grading (G1).  
\* 7.1.4. Variables:  
\* Independent: Grading condition (G1: Non-Rubric, G2: Rubric-Based).  
\* Dependent: STDEV of total scores per transcript, MAE of total scores from target, distribution of total scores.  
\* Controlled: LLM model, temperature, synthetic transcript content, number of grading attempts.  
\* 7.1.5. Materials & Setup:  
\* Transcript Set: 50 synthetic transcripts.  
\* Generation Process: For each of 50 iterations:  
1\. Randomly select a target total score (TTS) from a stratified distribution: 20% of scores in the range 5-10 (low), 40% in 11-15 (medium), and 40% in 16-20 (high). The scores within each stratum are chosen uniformly at random from that stratum's range.  
2\. Prompt `GPT-4.1` (temperature 0.7 for creative generation):  
\`\`\`  
You are an expert in simulating patient-doctor interactions for health communication research.  
Your task is to generate a patient-doctor dialogue transcript.  
The patient's communication proficiency in this transcript should realistically correspond to a TOTAL score of {{TTS}} when assessed using the provided "Patient Health Communication Rubric v5.0".  
First, provide a brief outline of the conversation flow and key patient utterances that would lead to this score.  
Then, generate the full patient-doctor dialogue transcript (approximately 300-700 words).  
Do NOT explicitly mention the rubric or the target score in the generated transcript itself.  
 Patient Health Communication Rubric v5.0:  
 \[Embed Full Rubric 5.0 Text Here \- as in Section 4\]

                Target Total Score for this transcript: {{TTS}}
                Outline:
                \[LLM generates outline\]
                Transcript:
                \[LLM generates transcript\]
                \`\`\`
            3\.  Store each generated transcript with its TTS and a unique SyntheticTranscriptID. Ensure the LLM does not include the rubric text or score in the final transcript output.
    \* LLM for Grading: \`gpt-4.1-mini\` (version as per Section 6). Temperature: 0.3.
    \* Rubric Version: Rubric 5.0.
    \* Prompt Templates for Grading:
        \* \*\*P1.1 (G1 \- Non-Rubric):\*\*
            \`\`\`
            You are a health communication analyst. Read the following patient-doctor transcript.
            Assess the patient's overall communication proficiency based on your general understanding of effective health communication.
            Assign a single holistic total score from 5 (very poor) to 20 (excellent).
            Output only the numerical total score. Example: 15

            Transcript:
            {{TranscriptText}}

            Total Score (5-20):
            \`\`\`
        \* \*\*P1.2 (G2 \- Rubric-Based):\*\*
            \`\`\`
            You are a health communication analyst. You must use the provided "Patient Health Communication Rubric v5.0" to assess the patient's communication in the following transcript.
            For each of the 5 categories in the rubric, provide a score from 1 to 4\.
            Then, provide a total score, which is the sum of the 5 category scores (must be between 5 and 20).
            Output the scores in the following exact format:
            Clarity of Language: \[score\_1\_4\]
            Lexical Diversity: \[score\_1\_4\]
            Conciseness and Completeness: \[score\_1\_4\]
            Engagement with Health Information: \[score\_1\_4\]
            Health Literacy Indicator: \[score\_1\_4\]
            Total Score: \[sum\_of\_scores\_5\_20\]

            Patient Health Communication Rubric v5.0:
            \[Embed Full Rubric 5.0 Text Here \- as in Section 4\]

            Transcript:
            {{TranscriptText}}

            Scores:
            \`\`\`

\* \*\*7.1.6. Procedure:\*\*  
 1\. Generate the 50 synthetic transcripts as described.  
 2\. For each of the 50 SyntheticTranscriptIDs:  
 a. Perform 50 grading attempts using Prompt P1.1 (G1). Log TranscriptID, attempt number, prompt, full LLM response, extracted total score.  
 b. Perform 50 grading attempts using Prompt P1.2 (G2). Log TranscriptID, attempt number, prompt, full LLM response, extracted category scores, extracted total score.  
 (Total grading attempts: 50 transcripts \* 2 conditions \* 50 attempts/condition \= 5000).  
\* \*\*7.1.7. Data Analysis Plan (Alpha \= 0.05 for all tests):\*\*  
 1\. For each transcript and condition, calculate STDEV of the 50 total scores.  
 2\. Compare the distributions of STDEVs between G1 and G2 using a Mann-Whitney U test (for H1a). Report medians and IQRs of STDEVs for G1 and G2.  
 3\. For each transcript and condition, calculate MAE \= |mean_LLM_total_score \- TTS|. Compare MAEs between G1 and G2 using a Wilcoxon signed-rank test (paired by transcript) (for H1b). Report medians and IQRs of MAEs. The MAE for G2 serves as a key indicator of how well the LLM adheres to the rubric's intended scoring when the rubric is explicitly provided.  
 4\. Plot histograms of total scores for G1, G2, and the original TTS distribution. Qualitatively compare alignment (for H1c). Consider Kullback-Leibler divergence if appropriate.  
 \*Rationale for G1 Baseline:\* Comparing against an unguided LLM (G1) provides a more informative baseline for the \*added value of Rubric 5.0\* than comparing against purely random scoring. If G2 significantly outperforms G1 in consistency (STDEV) and accuracy against rubric-derived targets (MAE), it demonstrates the rubric's utility in structuring the LLM's assessment beyond its general capabilities.\*  
\* \*\*7.1.8. Expected Output/Deliverables:\*\*  
 \* Dataset of 50 synthetic transcripts with their TTS.  
 \* Dataset of all 5000 grading attempts with scores.  
 \* Calculated STDEVs and MAEs per transcript/condition.  
 \* Statistical test results (p-values, test statistics).  
 \* Summary statistics and plots.

7.2. Experiment 2: Prompt Strategy Optimization  
\* 7.2.1. Experiment ID & Title: EXP2_PromptOptimization  
\* 7.2.2. Objective(s): Determine the optimal prompt engineering strategy for applying Rubric 5.0 to maximize scoring consistency.  
\* 7.2.3. Hypothesis(es): H2: Chain-of-Thought (CoT) prompting will result in significantly lower STDEV of total scores compared to zero-shot or few-shot prompting.  
\* 7.2.4. Variables:  
\* Independent: Prompting strategy (Zero-shot, Few-shot, CoT).  
\* Dependent: STDEV of total scores per transcript.  
\* Controlled: LLM model, temperature, transcript content (real transcripts from Set A), rubric version.  
\* 7.2.5. Materials & Setup:  
\* Transcript Set: Set A (N=50 real transcripts).  
\* LLM for Grading: gpt-4o-mini (version as per Section 6). Temperature: 0.1.  
\* Rubric Version: Rubric 5.0.  
\* Few-Shot Examples (N=3): Create 3 illustrative examples. Each example will contain: (a) a short, distinct (not from experimental sets) patient-doctor transcript snippet, (b) correctly assigned category scores and total score using Rubric 5.0, (c) brief rationale for scores. Example format:  
Example Transcript Snippet: \[Snippet Text\] Rubric 5.0 Scores: Clarity of Language: 3 Lexical Diversity: 2 Conciseness and Completeness: 3 Engagement with Health Information: 2 Health Literacy Indicator: 2 Total Score: 12 Rationale: \[Brief rationale linking snippet to scores\]  
\* Prompt Templates for Grading (all output formats same as P1.2):  
\* P2.1 (Zero-shot): Identical to P1.2 (Rubric-Based prompt from Exp 1).  
\* P2.2 (Few-shot):  
\`\`\`  
You are a health communication analyst. You must use the provided "Patient Health Communication Rubric v5.0" to assess the patient's communication.  
Here are some examples of how to apply Rubric 5.0:  
\[Embed Example 1: Snippet, Scores, Rationale\]  
\---  
\[Embed Example 2: Snippet, Scores, Rationale\]  
\---  
\[Embed Example 3: Snippet, Scores, Rationale\]  
\---  
Now, using "Patient Health Communication Rubric v5.0", assess the patient's communication in the following new transcript.  
Output the scores in the following exact format:  
Clarity of Language: \[score_1_4\]  
Lexical Diversity: \[score_1_4\]  
Conciseness and Completeness: \[score_1_4\]  
Engagement with Health Information: \[score_1_4\]  
Health Literacy Indicator: \[score_1_4\]  
Total Score: \[sum_of_scores_5_20\]  
 Patient Health Communication Rubric v5.0:  
 \[Embed Full Rubric 5.0 Text Here \- as in Section 4\]

            New Transcript:
            {{TranscriptText}}

            Scores:
            \`\`\`
        \* \*\*P2.3 (CoT \- Chain-of-Thought):\*\*
            \`\`\`
            You are a health communication analyst. You must use the provided "Patient Health Communication Rubric v5.0" to assess the patient's communication in the following transcript.
            Think step-by-step for each of the 5 rubric categories. For each category:
            1\. Briefly state your reasoning and cite evidence from the transcript that supports your score for this category.
            2\. Assign a score from 1 to 4 for this category.
            After scoring all 5 categories, calculate and provide the total score (sum of the 5 category scores, must be between 5 and 20).
            Structure your output clearly, providing reasoning THEN the score for each category, followed by the final list of scores in the specified format.

            Patient Health Communication Rubric v5.0:
            \[Embed Full Rubric 5.0 Text Here \- as in Section 4\]

            Transcript:
            {{TranscriptText}}

            Step-by-step Analysis and Scoring:
            \[LLM generates reasoning and scores per category here during its "thought" process\]

            Final Scores Output:
            Clarity of Language: \[score\_1\_4\]
            Lexical Diversity: \[score\_1\_4\]
            Conciseness and Completeness: \[score\_1\_4\]
            Engagement with Health Information: \[score\_1\_4\]
            Health Literacy Indicator: \[score\_1\_4\]
            Total Score: \[sum\_of\_scores\_5\_20\]
            \`\`\`
            \*(AI Executor Note: For CoT, the LLM will generate reasoning text before the final score block. Log the entire response. For STDEV calculation, only parse the final score block.)\*

\* \*\*7.2.6. Procedure:\*\*  
 1\. For each of the 50 TranscriptIDs in Set A:  
 a. Perform 25 grading attempts using Prompt P2.1 (Zero-shot).  
 b. Perform 25 grading attempts using Prompt P2.2 (Few-shot).  
 c. Perform 25 grading attempts using Prompt P2.3 (CoT).  
 Log all required data for each attempt.  
 (Total grading attempts: 50 transcripts \* 3 strategies \* 25 attempts/strategy \= 3750).  
\* \*\*7.2.7. Data Analysis Plan (Alpha \= 0.05):\*\*  
 1\. For each transcript and strategy, calculate STDEV of the 25 total scores.  
 2\. Compare STDEVs across the 3 strategies using Friedman's test.  
 3\. If Friedman's test is significant, perform post-hoc pairwise comparisons (e.g., Nemenyi test or Wilcoxon signed-rank tests with Bonferroni correction) to identify which strategies differ significantly. Report effect sizes (e.g., Cohen's d for paired comparisons if using t-test like approximations or rank-biserial correlation).  
\* \*\*7.2.8. Expected Output/Deliverables:\*\*  
 \* Dataset of all 3750 grading attempts.  
 \* Calculated STDEVs per transcript/strategy.  
 \* Statistical test results.  
 \* Identification of the "Winning Prompt Strategy" (lowest significant STDEV) to be used in Exp 3\.

7.3. Experiment 3: LLM Architecture Comparison  
\* 7.3.1. Experiment ID & Title: EXP3_ModelComparison  
\* 7.3.2. Objective(s): To compare the performance of different state-of-the-art LLM architectures in applying Rubric 5.0 with the optimized prompt strategy, focusing on scoring consistency, qualitative reasoning characteristics, and practical considerations, thereby establishing a methodology for ongoing model selection in this evolving landscape.  
\* 7.3.3. Hypothesis(es): H3: Different LLM architectures will exhibit significantly different STDEVs in total scores and qualitatively distinct reasoning patterns when applying the same optimized prompt strategy and rubric.  
\* 7.3.4. Variables:  
\* Independent: LLM Model (e.g., gpt-4o-mini, gpt-4o, claude-3-5-sonnet-20240620, gemini-1.5-pro-latest).  
\* Dependent: STDEV of total scores per transcript; Mean total scores per transcript (exploratory); Qualitative characteristics of reasoning (to be analyzed in Exp 4); API latency and estimated cost.  
\* Controlled: Prompt strategy (winning strategy from Exp 2), temperature, transcript content (real transcripts from Set B), rubric version.  
\* 7.3.5. Materials & Setup:  
\* Transcript Set: Set B (N=50 real transcripts, different from Set A).  
\* LLMs for Grading: Models listed in Section 6\. Temperature: 0.1 for all.  
\* Rubric Version: Rubric 5.0.  
\* Prompt Template: The "Winning Prompt Strategy" identified in Exp 2 (e.g., P2.3 if CoT wins).  
\* 7.3.6. Procedure:  
1\. For each of the 50 TranscriptIDs in Set B:  
a. For each LLM model being tested:  
i. Perform 20 grading attempts using the Winning Prompt Strategy.  
Log all required data, including the specific model version used for each attempt.  
(Total grading attempts: 50 transcripts \* 4 models \* 20 attempts/model \= 4000).  
\* 7.3.7. Data Analysis Plan (Alpha \= 0.05):  
1\. For each transcript and model, calculate STDEV of the 20 total scores.  
2\. Compare STDEVs across the models using Friedman's test (or repeated measures ANOVA if assumptions met for transformed STDEVs).  
3\. If significant, perform post-hoc pairwise comparisons. Report effect sizes.  
4\. Exploratory: For each transcript and model, calculate mean total score. Compare mean scores across models to check for systematic scoring differences.  
5\. Record average API call latency and estimated cost per model if available.  
\* 7.3.8. Expected Output/Deliverables:  
\* Dataset of all 4000 grading attempts.  
\* Calculated STDEVs and mean scores per transcript/model.  
\* Statistical test results comparing consistency across models.  
\* Comparative data on API latency and estimated costs.  
\* Characterization of current leading models based on empirical data and a recommended framework/methodology for selecting and evaluating LLMs for this task as new models become available or existing ones are updated.

7.4. Experiment 4: Qualitative Reasoning Analysis  
\* 7.4.1. Experiment ID & Title: EXP4_ReasoningAnalysis  
\* 7.4.2. Objective(s): To qualitatively characterize and compare the reasoning patterns of different LLMs (from Exp 3) when applying Rubric 5.0 using the optimized (CoT-style if selected) prompt, specifically assessing alignment with rubric criteria, evidence citation, and identifying potential issues like hallucinations or superficial reasoning (Lit Review: Sec 5.3, Sources 5, 33, 15, 18).  
\* 7.4.3. Hypothesis(es): (More exploratory, less formal hypotheses)  
\* H4a: Different LLMs will exhibit qualitatively different reasoning styles (e.g., reliance on direct quotes vs. paraphrasing, level of detail, depth of rubric interpretation).  
\* H4b: Some LLMs may demonstrate reasoning that aligns more closely with the intended interpretation of Rubric 5.0 criteria, while others might show tendencies towards issues like hallucination of evidence or superficial justification.  
\* 7.4.4. Variables:  
\* Independent: LLM Model (from Exp 3).  
\* Dependent: Coded characteristics of LLM-generated reasoning.  
\* Controlled: Transcripts, prompt (CoT-style), rubric.  
\* 7.4.5. Materials & Setup:  
\* Reasoning Data: Full LLM responses (including step-by-step reasoning if CoT was used) from Exp 3\.  
\* Transcript Subset: Select 15 diverse transcripts from Set B (used in Exp 3). Diversity criteria: range of initial STDEVs from Exp 3, varying lengths, different apparent communication qualities.  
\* Reasoning Coding Scheme (to be developed by human researchers, then applied by AI if feasible or by humans):  
\* Initial Pass: Human researchers review reasoning for 5 transcripts across all models to draft initial coding categories.  
\* Categories may include:  
1\. Evidence Type: (Direct Quote, Paraphrase, General Gist, No Clear Evidence)  
2\. Rubric Alignment: (Clear link to rubric definition, Loose link, Misinterpretation of rubric)  
3\. Specificity of Reasoning: (Specific to transcript details, Generic statement)  
4\. Reasoning Depth: (Superficial, Moderate, In-depth)  
5\. Identification of Strengths/Weaknesses: (Balanced, Focuses on one, Misses obvious points)  
\* Refinement: Two human researchers independently code reasoning for 3 transcripts using the draft scheme. Calculate Inter-Coder Reliability (ICR) using Cohen's Kappa. Refine coding scheme definitions until Kappa \>= 0.70.  
\* 7.4.6. Procedure (AI applies final coding scheme if possible, or flags for human coding):  
1\. For each of the 15 selected transcripts:  
a. For each LLM model's reasoning output from Exp 3:  
i. Segment reasoning pertaining to each of the 5 rubric categories.  
ii. Apply the finalized Reasoning Coding Scheme to each segment. Log coded categories.  
\* 7.4.7. Data Analysis Plan:  
1\. Calculate frequencies of each reasoning code per LLM model, aggregated across transcripts and rubric categories.  
2\. Use Chi-squared tests or Fisher's exact tests to compare distributions of codes across models.  
3\. Qualitatively summarize distinct reasoning patterns, strengths, and weaknesses for each LLM, supported by examples.  
\* 7.4.8. Expected Output/Deliverables:  
\* Finalized Reasoning Coding Scheme.  
\* Dataset of coded reasoning segments.  
\* ICR for human coders (if applicable for scheme development).  
\* Statistical comparisons of code frequencies.  
\* Qualitative report summarizing reasoning styles per LLM, to inform selection of "Winning LLM."  
7.5. Experiment 5: Optimized Tool Validation  
\* 7.5.1. Experiment ID & Title: EXP5_FinalToolValidation  
\* 7.5.2. Objective(s): 1. To assess the scoring consistency (reliability) of the final, optimized LLM assessment tool configuration on a large, unseen set of real transcripts. 2. To explore and report on feasible methods for quantifying the uncertainty/confidence of the optimized tool's scores, providing an initial characterization of these properties (Lit Review: Sec 5.2).

- 7.5.3. Hypothesis(es): (Benchmark setting)
  - H5: The final optimized tool will demonstrate a mean STDEV of total scores below a pre-defined threshold (e.g., <0.5 for individual category scores on 1-4 scale, implying < sqrt(5\*0.5^2) approx <1.12 for total score 5-20; or more stringently, aim for overall total score STDEV < 1.0, aligning with Success Metric 2).
  - Exploratory Goal: To characterize the confidence levels of the LLM's scores and identify if/when the model exhibits higher uncertainty.
- 7.5.4. Variables:
- Independent: N/A (single best tool configuration).
- Dependent: STDEV of total scores and per-category scores; Distribution of scores; Measures of score uncertainty/confidence (if quantifiable).
- Controlled: LLM model (Winning LLM from Exp 3, informed by Exp 4), prompt strategy (Winning Prompt from Exp 2), temperature, rubric version.
- 7.5.5. Materials & Setup: \* Transcript Set: Set C (N=approx. 577 real transcripts, unseen in Exp 2 & 3).  
  \* Optimized Tool Configuration:  
  \* LLM: "Winning LLM Model" (e.g., claude-3-5-sonnet-20240620 if it performed best).  
  \* Prompt: "Winning Prompt Strategy" (e.g., P2.3 CoT prompt).  
  \* Temperature: 0.1. Rubric: Rubric 5.0.  
  \* 7.5.6. Procedure:  
  1\. For each of the TranscriptIDs in Set C:  
  a. Perform 10 grading attempts using the Optimized Tool Configuration.  
  Log all required data.  
  (Total grading attempts: approx. 577 transcripts \* 10 attempts \= 5770).  
  \* 7.5.7. Data Analysis Plan:  
  1\. For each transcript, calculate STDEV of the 10 total scores and STDEV for each of the 5 category scores.  
  2\. Report mean, median, IQR, and distribution (histogram) of these STDEVs across all transcripts in Set C. Compare against Success Metric 2.  
  3\. Calculate the percentage of transcripts achieving STDEV < [thresholds, e.g., 0.3, 0.5 for category scores].  
  4\. Report descriptive statistics for the actual scores assigned (mean, median, range for total and category scores).  
  5\. Uncertainty Quantification: Explore and apply feasible methods to assess score uncertainty/confidence. Candidate methods include (Lit Review: Sec 5.2, Source 51):  
   _ Analyzing output token probabilities for scores or key reasoning phrases, if accessible via the API.  
   _ Running a small number of additional grading attempts per transcript with slight variations in temperature or by sampling multiple outputs if the API supports n-best responses, then examining score variance.  
   _ If using an ensemble approach (not planned but a general method): variance across ensemble member scores.  
   _ (Not directly feasible for this project but for context: Entropy-based measures, Bayesian approaches, MC Dropout are more advanced techniques typically requiring deeper model access or specific architectures).  
   Focus on practical methods given API limitations. Report on observed confidence characteristics (e.g., are lower-confidence scores associated with transcripts that also have higher STDEV?).  
  6\. Exploratory: If transcript metadata is available (e.g., length, source details if Set C mixes sources), analyze if STDEV or uncertainty varies by these factors using appropriate statistical tests (e.g., ANOVA, correlation).  
  \* 7.5.8. Expected Output/Deliverables:  
  \* Dataset of all ~5770 grading attempts.  
  \* Calculated STDEVs per transcript (total and category).  
  \* Summary statistics and plots of STDEVs and scores.  
  \* Benchmark performance metrics for the final optimized tool against pre-defined success criteria.  
  \* Report on the exploration of uncertainty/confidence quantification methods and initial findings regarding the tool's score confidence characteristics.

**8\. Overall Data Management & Analysis**

- **Centralized Data Storage:** All raw API call logs, parsed scores, reasoning text, calculated metrics (STDEVs, MAEs, correlations), statistical test outputs (p-values, test statistics, confidence intervals), calibration metrics (e.g., ECE, Brier score if uncertainty is quantified), and generated figures will be stored in a structured, version-controlled repository (e.g., dedicated cloud storage bucket or Git LFS-managed repository). Define clear directory structures and file naming conventions.
  - Example: /project_root/data/exp1/raw_api_logs/, /project_root/data/exp1/processed_scores/, /project_root/results/exp1/stats/
- **Data Schema:** For primary data tables (e.g., individual grading attempts), define schema: AttemptID, ExperimentID, TranscriptID, ConditionID (prompt/model), AttemptNum, Timestamp, LLM_Model_Version, FullRequestPrompt, FullLLM_Response, Parsed_Score_Cat1, ..., Parsed_Score_Cat5, Parsed_Score_Total, Parsed_Reasoning_Text (if applicable), LLM_Output_Confidence_Score (if retrievable/calculable), Cost, API_Latency, Error_Flag, Error_Message.
- **Analysis Code:** All scripts for data processing, statistical analysis (including reliability measures like Kappa, ICC where appropriate for inter-LLM comparisons or against synthetic targets, and calibration metrics from Lit Review Table 3), and figure generation will be written in Python (using libraries like Pandas, NumPy, SciPy, Statsmodels, Matplotlib, Seaborn) and version controlled (e.g., in a Git repository). Notebooks (e.g., Jupyter) can be used for exploratory analysis but final analysis pipelines should be script-based for reproducibility.

**9\. Paper Outline (Targeting JAMA-style Original Research Article)**

- **Title:** (To be refined based on final key findings, e.g., "A Systematic Evaluation of Large Language Model Configurations for Rubric-Based Assessment of Patient Health Communication Proficiency")
- **Abstract:** (Structured: Importance, Objective, Design, Setting, Participants, Main Outcome(s) and Measure(s), Results, Conclusions and Relevance) \- Reflecting findings from Exp 1-5.
- **Key Points:** (3 bullet points: Question, Findings, Meaning)
- **Introduction:**
  - 1.1 Background on importance of health communication, challenges in assessment.
  - 1.2 Potential of LLMs, need for rigorous methodology, statement of project aim.
- **Methods:**
  - 2.1 Rubric Development (Patient Health Communication Rubric v5.0 \- brief description and justification).
  - 2.2 Transcript Datasets (Sources, de-identification, partitioning).
  - 2.3 Experimental Design Overview (mentioning the sequence of optimization and validation experiments).
  - 2.4 LLM Models and Prompting Strategies Investigated.
  - 2.5 For each key experimental phase (derived from Exp 1-5):
    - Objective
    - Key procedures
    - Outcome measures (e.g., STDEV, MAE)
    - Statistical analysis approaches
  - 2.6 Qualitative Reasoning Analysis Methodology (Exp 4).
- **Results:**
  - 3.1 Baseline LLM-Rubric Utility and Synthetic Transcript Performance (Exp 1 findings).
  - 3.2 Prompt Strategy Optimization (Exp 2 findings \- identify winning strategy).
  - 3.3 LLM Architecture Comparison (Exp 3 findings \- identify winning model).
  - 3.4 Qualitative Analysis of LLM Reasoning (Exp 4 findings \- characterize reasoning of different models).
  - 3.5 Performance of the Final Optimized Assessment Tool (Exp 5 findings \- benchmark consistency).
- **Discussion:**
  - 4.1 Summary of Key Findings (e.g., optimal LLM configuration, achieved consistency).
  - 4.2 Interpretation of Findings (why certain strategies/models performed better, implications of reasoning patterns).
  - 4.3 Strengths of the Study (methodological rigor, systematic comparisons).
  - 4.4 Limitations (scope of transcripts, specific LLMs tested, consistency vs. accuracy, potential for LLM drift over time).
  - 4.5 Comparison with Existing Literature (if any).
  - 4.6 Potential Applications and Use Cases (from Outline 2.1).
  - 4.7 Future Directions (including human expert validation, real-world implementation studies, multilingual adaptation).
- **Conclusions:** Succinct summary of the main takeaway regarding the developed LLM assessment framework.
- **References**
- **Figures (Max 5 for main paper, others as supplementary):**
  - Fig 1: Comparison of STDEVs for Prompt Strategies (Exp 2).
  - Fig 2: Comparison of STDEVs for LLM Architectures (Exp 3).
  - Fig 3: Distribution of STDEVs for the Final Optimized Tool (Exp 5).
  - Fig 4: Qualitative examples of LLM reasoning differences (Exp 4\) or MAE for synthetic transcripts (Exp 1).
  - Fig 5: TBD based on most impactful visual data.
- **Tables:**
  - Table 1: Descriptive statistics of transcript datasets.
  - Table 2: Key statistical results for prompt/model comparisons.
  - Table 3: Frequencies of reasoning codes (Exp 4).
- **Supplementary Materials:** Full Rubric 5.0, detailed prompt texts, additional figures/tables, list of all transcripts (IDs only).

10\. Ethical Considerations Statement (for the paper)

This statement should be significantly expanded based on the comprehensive discussion in the `extensive-lit-review.md` (Section 6). It should address the following key areas, tailored to the specifics of this project:

- **Data Privacy and Security:** Detail adherence to HIPAA/GDPR principles for the de-identified datasets used. Discuss the de-identification methods applied to the source datasets and acknowledge the residual risks and challenges in de-identifying conversational data (balancing privacy with data utility for LLM analysis). Emphasize that no re-identification will be attempted. (Lit Review: Sec 6.1, Sources 30, 56, 57).
- **Algorithmic Bias and Health Equity:** State the project's commitment to promoting health equity. Acknowledge the potential for LLMs to inherit or amplify biases from training data. Describe any steps taken during model selection or prompt design to explore or mitigate potential biases related to patient communication styles (e.g., associated with dialect, culture, or demographics, if such metadata were available and ethically usable for fairness checks). Note that formal bias audits across demographic groups are limited by available metadata in the current datasets but are crucial for future work and broader deployment. (Lit Review: Sec 6.2, Sources 13, 50, 38).
- **Patient Autonomy and Informed Consent (Contextual):** Since this project uses pre-existing, de-identified datasets, new patient consent for _this specific research_ is not applicable. However, the statement should briefly touch upon the importance of informed consent and transparency when such AI tools are deployed in live clinical settings, noting the complexities of consent for AI-driven analysis of personal conversations. (Lit Review: Sec 6.3, Sources 14, 61).
- **Accountability, Transparency, and Intended Use:** Clarify that the developed framework is a research prototype intended to assess patient communication proficiency based on a rubric, not a diagnostic tool for clinical decision-making regarding individual patient care. Note the importance of transparency in LLM operations (e.g., through CoT prompting and qualitative analysis of reasoning) and the need for human oversight in any future clinical application. (Lit Review: Sec 6.4, Sources 14, 58, 62).
- **Regulatory Considerations (Future Outlook):** Briefly mention that if such a tool were to be developed into a product for clinical use (beyond this research project), it might be considered Software as a Medical Device (SaMD) and would require appropriate regulatory review (e.g., by the FDA) to ensure safety and effectiveness. (Lit Review: Sec 6.4, Sources 63, 64).

Example text structure (to be adapted and expanded based on final project details):
"This research utilized de-identified patient-doctor transcripts from [Source 1 details] and [Source 2 details], obtained under documented ethical approvals or waivers from the original data collectors. All data were handled in accordance with privacy principles aligning with HIPAA and GDPR. The de-identification process applied to these datasets aimed to protect patient privacy, though the inherent challenges of fully balancing anonymization with the preservation of conversational richness for AI analysis are acknowledged (Lit Review: Sources 30, 31). This study did not involve direct patient interaction or new data collection, thus direct informed consent for this specific secondary analysis was not applicable. We recognize the critical importance of addressing potential algorithmic biases. While the current study is limited in its ability to perform exhaustive bias audits due to metadata constraints in the provided datasets, fairness considerations informed our qualitative analyses and are highlighted as essential for future development and deployment of such AI tools (Lit Review: Sec 6.2). The LLM-based assessment framework developed herein is a research tool intended for evaluating communication proficiency against a rubric and is not designed or validated for direct clinical decision-making. Any future transition to clinical use would necessitate human oversight, further validation, and appropriate regulatory consideration (Lit Review: Sec 6.4)."

11\. Reproducibility Statement (for the paper)

"The Patient Health Communication Rubric v5.0, detailed prompt templates used for LLM interactions, and code for data analysis are available in the supplementary materials and/or at \[Link to Public Repository, e.g., GitHub, OSF\]. Due to data use agreements, the full transcripts cannot be publicly shared, but Transcript IDs corresponding to publicly available sources are provided. LLM models and versions used are detailed in the Methods section. We aim to provide sufficient detail to allow for replication of our methodology."  
**12\. Contingency Planning**

- **LLM API Unavailability/Deprecation:** If a primary selected LLM becomes unavailable or significantly changes during the project:
  1. Pause experiments involving that model.
  2. Evaluate suitable alternative models with similar capabilities, referring to the model selection framework developed in Exp 3.
  3. If an alternative is chosen, document the change and rationale. May require re-running a small bridging experiment to compare the alternative to previously tested models on a small subset of data.
- **Unexpectedly High API Costs:** Monitor costs closely during pilots. If projected costs exceed budget, consider:
  1. Reducing the number of grading attempts per transcript (if initial STDEVs are already very low and stable, and sufficient for statistical power).
  2. Prioritizing the most promising models/prompts for more extensive testing, potentially reducing the scope of less critical comparisons.
- **Results Deviate Significantly from Hypotheses (e.g., CoT performs worse, high STDEVs persist):**
  1. Double-check implementation, prompts, data logging, and analysis scripts for errors.
  2. Investigate potential reasons for deviation: Are prompts misunderstood? Is the task too complex for current models with the chosen approach? Are there inherent ambiguities in the rubric or transcripts affecting consistency?
  3. If results are robust, proceed with analysis and report findings as observed. This is still valuable scientific information that can inform future research.
  4. Re-evaluate assumptions and discuss potential reasons for deviation in the paper.
- **Challenges with LLM Behavior (Hallucinations, Reasoning Flaws, Bias):**
  1.  _Detection:_ Experiment 4 (Qualitative Reasoning) is designed to help detect such issues. Score inconsistencies or unexpected patterns in Exp 5 might also indicate problems.
  2.  _Mitigation/Management during project:_ If significant hallucinations or reasoning flaws are tied to a specific model or prompt, attempt prompt refinement. If pervasive, document limitations thoroughly. If bias is suspected (e.g., from qualitative review), note this limitation and the need for diverse data/audits in future work.
  3.  _Reporting:_ Transparently report any observed limitations or biases in the final manuscript. (Lit Review: Sec 5.1, 5.3, 6.2).
- **Error Handling by AI Executor:**
  - Implement retry logic for transient API errors (e.g., rate limits, temporary server issues) with exponential backoff, up to 3 retries.
  - Log all errors meticulously, including error messages and context.
  - If a specific transcript consistently causes errors for a model/prompt, flag it for investigation and potentially exclude it from that specific analysis if unresolvable, documenting the exclusion.
