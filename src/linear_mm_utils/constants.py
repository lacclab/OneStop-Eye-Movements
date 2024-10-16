import pandas as pd
import numpy as np

colors = [
    "#347BB9",
    "#AE0135",
    "#003366",
    "darkolivegreen",
    "darkcyan",
    "darkslategray",
    "darkslateblue",
    "turquoise",
    "darkslategray",
    "cadetblue",
    "steelblue",
    "royalblue",
    "midnightblue",
]
re_columns = ["unique_paragraph_id", "subject_id"]

# ignore pandas warnings
pd.options.mode.chained_assignment = None  # default='warn'
# set seeds
np.random.seed(42)


outcome_name_mapping = {
    "IA_DWELL_TIME": "Total Fixation Duration",
    "IA_FIRST_FIXATION_DURATION": "First Fixation Duration",
    "IA_FIRST_PASS_GAZE_DURATION": "First Pass Gaze Duration",
    "IA_FIRST_RUN_DWELL_TIME": "Gaze Duration",
    "IA_RUN_COUNT": "Fixation Count",
    "NEXT_SAC_AMPLITUDE": "Saccade Length",
    "PREVIOUS_SAC_AMPLITUDE": "Saccade Length",
    "IA_REGRESSION_OUT_FULL_COUNT": "Regression Rate",
    "IA_SKIP": "Skip Rate",
    "IA_SINGLE_FIX_DURATION": "Single Fixation Duration",
    "IA_FIXATION_COUNT": "Fixation Count",
    "IA_ZERO_TF": "Global Skip Rate (TF = 0)",
    "IA_DWELL_TIME_pos": "Total Fixation Duration (> 0)",
    "IA_FIRST_FIXATION_DURATION_pos": "First Fixation Duration (> 0)",
    "IA_FIRST_PASS_GAZE_DURATION_pos": "First Pass Gaze Duration (> 0)",
    "IA_FIRST_RUN_DWELL_TIME_pos": "Gaze Duration (> 0)",
    "CURRENT_FIX_DURATION": "Single Fixation Duration",
    "IS_NEXT_REGRESSION": "Regression Rate",
    "IS_PREV_REGRESSION": "Regression Rate",
}

# add to outcome_name_mapping "outcome_name_zscore" for each outcome
zscore_name_mapping = {}
for outcome in outcome_name_mapping.keys():
    zscore_name_mapping[outcome + "_zscore"] = (
        outcome_name_mapping[outcome] + " (z-score)"
    )

outcome_name_mapping.update(zscore_name_mapping)

outcome_units_mapping = {
    "IA_DWELL_TIME": "(ms)",
    "IA_FIRST_FIXATION_DURATION": "(ms)",
    "IA_FIRST_PASS_GAZE_DURATION": "(ms)",
    "IA_FIRST_RUN_DWELL_TIME": "(ms)",
    "IA_RUN_COUNT": "",
    "IA_REGRESSION_OUT_FULL_COUNT": "(count)",
    "IA_SKIP": "",
    "IA_ZERO_TF": "",
    "IA_DWELL_TIME_pos": "ms",
    "IA_FIRST_FIXATION_DURATION_pos": "ms",
    "IA_FIRST_PASS_GAZE_DURATION_pos": "ms",
    "IA_FIRST_RUN_DWELL_TIME_pos": "ms",
    "CURRENT_FIX_DURATION": "ms",
}

# do the same with outcome_units_mapping
zscore_units_mapping = {}
for outcome in outcome_units_mapping.keys():
    zscore_units_mapping[outcome + "_zscore"] = outcome_units_mapping[outcome]

outcome_units_mapping.update(zscore_units_mapping)


explanatory_var_class_names = {
    "reread": {
        0: "First Reading",
        1: "Repeated Reading",
        "0": "First Reading",
        "1": "Repeated Reading",
    },
    # "has_preview_reread": {
    #     "Gathering_1": "Gathering, RR",
    #     "Gathering_0": "Gathering, Not RR",
    #     "Hunting_1": "Hunting, RR",
    #     "Hunting_0": "Hunting, Not RR",
    # },
}

explanatory_var_names = {
    "has_preview": "Condition",
    "relative_to_aspan": "Relative to Critical Span",
    "article_ind_of_first_reading": "Article Index of the First Reading",
    "first_second_reading_types": "First (<10, 10) and Second (11, 12) Readings",
    "article_ind_group_of_first_reading": "Article Index of the First Reading",
    "reread": "Reread",
}

all_outcomes = [
    "IA_FIRST_FIXATION_DURATION",
    "IA_FIRST_RUN_DWELL_TIME",
    "IA_FIRST_PASS_GAZE_DURATION",
    "IA_RUN_COUNT",
    "IA_ZERO_TF",
    "IA_REGRESSION_OUT_FULL_COUNT",
]
