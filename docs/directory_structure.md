# Directory Structure

## data_[version]

<!-- ![Trial GIF](trial.gif)

:::{image} trial.gif
:alt: Trial GIF
:width: 400px -->

SR DataViewer Interest Area and Fixation Reports, and syntactic annotations.

- `sent_ia.tsv` Interest Area report.  
- `sent_fix.tsv` Fixations report.
- `annotations/` Syntactic annotations.

## participant_metadata

- `metadata.tsv` metadata on participants.
- `languages.tsv` information on languages spoken besides English.
- `test_scores/`
  - `test_conversion.tsv` unofficial conversion table between standardized proficiency tests (used to convert TOEIC to TOEFL scores).
  - `michigan-cefr.tsv` conversion table between form B and the newer forms D/E/F, as well as to CEFR levels.
  - `michigan/` item level responses for the Michigan Placement Test (MPT).
  - `comprehension/` item level responses for the reading comprehension during the eyetracking experiment.  

## splits

Trial and participant splits.

- `trials/`
  - `all_trials.txt` trial numbers for all the sentences (1-157).
  - `shared_trials.txt` trial numbers of the Shared Text regime.
  - `individual_trials.txt` trial number of the Individual Text regime.
- `participants/[version]/`
  - `random_order.csv` random participant order.
  - `train.csv` train participants.
  - `test.csv` test participants.

## dataset_analyses.Rmd

Analyses for the paper "".
Note that this script requires:
