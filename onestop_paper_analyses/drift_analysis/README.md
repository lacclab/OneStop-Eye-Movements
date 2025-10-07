# Drift Analysis

This folder contains the accuracy analysis of fixation assignments to lines in the OneStop eye-tracking dataset, quantifying the prevalence of erroneous line assignments due to vertical drift.

## Annotation Scheme

The annotation scheme (presented in the paper) classifies fixations into three categories:

1. **Line-reading fixations assigned to the correct line**
2. **Line-reading fixations assigned to an incorrect line** (drift errors)
3. **Other** - including:
   - Return sweep completion sequences
   - Fixations outside interest areas
   - Fixations shorter than 50ms or longer than 500ms

## Contents

- **`final_drift_annotations.csv`** - Manual annotations by two independent annotators
  - Contains fixation category annotations for 20 trials (1,296 fixations total)
  - Each trial is from a different randomly chosen participant in ordinary reading regime
  - Trials selected as the one with the smallest number of fixations among 10-line trials

- **`stats.ipynb`** - Statistical analysis notebook
  - Calculates inter-annotator agreement: Cohen's Kappa = 0.95
  - Computes error rates: 3.7% of fixations marked as incorrect line assignments by both annotators
  - Shows 72.5% line-reading fixations and 23.8% other fixations on average

- **`20scanpaths_shortest_10lines/`** - PDF visualizations of the 20 annotated trials
  - Fixations are numbered and color-coded by their assigned line
  - Shows horizontal lines indicating text line boundaries
  - Discarded fixations (< 50ms or > 500ms) shown in blue
