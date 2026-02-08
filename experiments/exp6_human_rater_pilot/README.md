# Experiment 6: Human Rater Pilot Study

## Overview

This experiment provides exploratory evidence for concurrent validity by comparing CHAT scores against human expert ratings on a subset of transcripts.

## Study Design

### Sample Size & Selection
- **Target**: 30-40 transcripts
- **Stratification**: Sample transcripts across CHAT score distribution (low, medium, high)
- **Data sources**: Include both DoVA and Nature transcripts
- **Selection criteria**: Transcripts with highest CHAT consistency (low STDEV) to ensure reliable LLM baseline

### Rating Protocol

#### Primary Outcomes (Option 1: Oral Health Literacy Focus)
Raters will score transcripts on a 1-4 scale for:

1. **Engagement with Health Information** (Primary outcome #1)
   - 1 (Poor): Brief responses, no elaboration
   - 2 (Fair): Adequate responses, minimal detail  
   - 3 (Good): Detailed responses, demonstrates understanding
   - 4 (Excellent): Detailed with questions, supplements information

2. **Overall Health Literacy** (Primary outcome #2)
   - 1 (Poor): Lacks understanding of basic health concepts
   - 2 (Fair): Limited understanding, requires simplification
   - 3 (Good): Basic understanding, uses relevant terms
   - 4 (Excellent): Strong understanding, clear articulation

**Note**: We intentionally limit to 2 outcomes to keep rater burden manageable and focus on core oral health literacy construct.

### Rater Recruitment
- **Target**: 2-3 raters minimum
- **Qualifications**: Clinicians, health literacy researchers, or trained research assistants with clinical communication background
- **Training**: 30-minute orientation including rubric review + 3 practice transcripts with discussion

### Statistical Analysis Plan

1. **Human Inter-Rater Reliability**
   - Intraclass Correlation Coefficient (ICC) - two-way random effects
   - Target: ICC > 0.60 (good agreement)
   
2. **Human-CHAT Agreement**
   - Spearman's correlation between mean human scores and CHAT scores
   - ICC for absolute agreement  
   - Target: ρ > 0.50 (moderate correlation indicates concurrent validity)

3. **Score Distribution Comparison**
   - Compare mean scores (humans vs CHAT)
   - Bland-Altman plots to visualize agreement

### Timeline
- **Week 1**: Finalize protocol, recruit raters, prepare materials
- **Week 2**: Rater training + pilot ratings (5 transcripts)
- **Week 3**: Main rating session (remaining transcripts)
- **Week 4**: Analysis and reporting

## Expected Outcomes

### Best Case
- Human ICC > 0.70 (good reliability)
- Human-CHAT ρ > 0.60 (strong correlation)
- Similar score distributions
- **Claim**: "Exploratory pilot shows CHAT scores correlate strongly with human expert ratings"

### Realistic Case
- Human ICC 0.50-0.70 (moderate reliability)
- Human-CHAT ρ 0.40-0.60 (moderate correlation)
- **Claim**: "Exploratory pilot shows CHAT scores correlate moderately with human ratings, supporting preliminary validity"

### Challenge Case
- Human ICC < 0.50 (poor reliability) OR Human-CHAT ρ < 0.40
- **Interpretation**: Human rating task may need refinement, or construct needs clearer definition
- **Still publishable**: Report honestly, discuss as limitation and direction for future work

## IRB Considerations

### Minimal Risk Determination
- Use existing de-identified transcripts (no new patient data collection)
- Raters assess transcripts, not patients directly
- Likely qualifies for exempt or expedited review

### Required Documentation
- Protocol summary
- Rater consent/information sheet (if required)
- Data security plan

## Materials to Prepare

1. **Rater Manual** (5-10 pages)
   - Rubric definitions with examples
   - Rating scale instructions
   - Practice transcripts with reference scores

2. **Rating Interface**
   - Simple spreadsheet or web form
   - Transcript ID + 2 rating fields (Engagement, Overall HL)
   - Space for optional comments

3. **Transcripts** (anonymized, formatted)
   - Remove file metadata
   - Consistent formatting
   - Random ordering to prevent bias

## Budget Considerations

### Rater Compensation
- Research assistants: May be part of existing research duties
- External clinicians: ~$50-100/hour × 3-4 hours = $150-400 per rater
- Total for 3 raters: ~$450-1,200

### Time Investment
- Principal investigator: ~10-15 hours (design, training, analysis)
- Each rater: ~3-4 hours (training + rating ~30-40 transcripts)

## Integration with Main Paper

### Methods Section
- Add subsection: "Exploratory Human Validation Pilot"
- Describe sampling, rating protocol, statistical analysis

### Results Section
- Report human ICC + Human-CHAT correlations
- Include Bland-Altman plot or correlation scatter
- 1-2 paragraphs maximum

### Discussion
- Frame as "preliminary evidence" or "exploratory validation"
- Note limitations (small sample, 2 outcomes only)
- Call for larger validation studies

## Files in This Directory

- `00_sample_selection.py` - Script to select stratified sample of transcripts
- `01_prepare_rating_materials.py` - Generate rater manual and rating sheets
- `02_analyze_human_ratings.py` - Compute ICC, correlations, create visualizations
- `rater_manual.md` - Complete rater training manual
- `rating_sheet_template.csv` - Template for raters to fill in

## Status

**Current Status**: Protocol designed, ready for implementation

**Next Steps**:
1. Run sample selection script to identify 30-40 transcripts
2. Prepare rater materials
3. Recruit raters
4. Execute pilot
5. Analyze and integrate into paper

