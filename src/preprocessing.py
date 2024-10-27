import json
import os
from pathlib import Path

import config
import numpy as np
import pandas as pd
import utils

# Run the script from the directory where it is located
script_directory = Path(__file__).parent
os.chdir(script_directory)


def create_full_report(df, metadata, survey):
    columns = [
        "RECORDING_SESSION_LABEL",
        "trials",
        "Trial_Index_",
        "session_interruptions_(recalibrations)",
        "total_recalibrations",
        "num_trials_without_recalibrations",
        "session_duration",
        "total_duration",
        "DURATION",
        "START_TIME",
        "END_TIME",
        "EYE_TRACKED",
        "total_blinks",
        "blinks_per_min",
        "total_average_blink_duration",
        "BLINK_COUNT",
        "AVERAGE_BLINK_DURATION",
        "batch_condition",
        "batch",
        "list",
        "article_id",
        "article_ind",
        "level",
        "level_ind",
        "paragraph_id",
        "q_ind",
        "question",
        "a",
    ]
    data_verification_report = pd.DataFrame(columns=columns)

    # find out unique participant
    unique_participant = np.unique(df["RECORDING_SESSION_LABEL"].values)

    # iterate over each trial
    for i, unique_part in enumerate(unique_participant):
        # from each unique participant get the 'block' from the original dataframe
        block = df.loc[df["RECORDING_SESSION_LABEL"] == unique_part]

        new_row = block.loc[block.index.values[0]].copy()

        total_recalibrations = block["RECALIBRATE"].sum()
        t, c = np.unique(block["trial"].values, return_counts=True)
        interruptions = zip(t, c)
        session_interruptions = 0
        for i in interruptions:
            if i[1] > 1:
                session_interruptions = session_interruptions + 1
        num_trials = block["INDEX"].max() - total_recalibrations
        start_time = block["START_TIME"].min() / 1000 / 60
        end_time = block["END_TIME"].max() / 1000 / 60
        session_duration = block["DURATION"].sum() / 1000 / 60
        total_blinks = block["BLINK_COUNT"].sum()
        total_average_blink_duration = (
            block["AVERAGE_BLINK_DURATION"].sum() / num_trials
        )

        mini_block = block[
            (block["reread"] == 0)
            & (block["practice"] == 0)
            & (block["RECALIBRATE"] == False)
        ]

        new_row["trials"] = mini_block["trial"].values.tolist()
        new_row["Trial_Index_"] = mini_block["Trial_Index_"].values.tolist()
        new_row["AVERAGE_BLINK_DURATION"] = mini_block[
            "AVERAGE_BLINK_DURATION"
        ].values.tolist()
        new_row["BLINK_COUNT"] = mini_block["BLINK_COUNT"].values.tolist()
        new_row["DURATION"] = mini_block["DURATION"].values.tolist()
        new_row["START_TIME"] = start_time
        new_row["END_TIME"] = end_time
        eye_tracked = np.unique(mini_block["EYE_TRACKED"].values)
        if len(eye_tracked) == 1:
            eye_tracked = eye_tracked[0][0]
        else:
            eye_tracked = "LR"
        new_row["EYE_TRACKED"] = eye_tracked
        new_row["total_recalibrations"] = total_recalibrations
        new_row["session_interruptions_(recalibrations)"] = session_interruptions
        new_row["session_duration"] = session_duration
        new_row["total_duration"] = end_time - start_time
        new_row["total_blinks"] = total_blinks
        new_row["blinks_per_min"] = total_blinks / session_duration
        new_row["total_average_blink_duration"] = total_average_blink_duration
        new_row["article_id"] = mini_block["article_id"].values.tolist()
        new_row["article_ind"] = mini_block["article_ind"].values.tolist()
        new_row["level"] = mini_block["level"].values.tolist()
        new_row["level_ind"] = mini_block["level_ind"].values.tolist()
        new_row["paragraph_id"] = mini_block["paragraph_id"].values.tolist()
        new_row["question"] = mini_block["question"].values.tolist()
        new_row["a"] = mini_block["a"].values.tolist()
        new_row["q_ind"] = mini_block["q_ind"].values.tolist()
        new_row["num_trials_without_recalibrations"] = mini_block["INDEX"].count()
        new_row["RECORDING_SESSION_LABEL"] = unique_part

        data_verification_report.loc[data_verification_report.shape[0]] = new_row

    comprehension_score = utils.get_comprehension_score(df)
    data_verification_report_with_score = utils.add_comprehension_score_to_df(
        data_verification_report, comprehension_score
    )
    data_verification_report_with_metadata = utils.add_metadata(
        data_verification_report_with_score, metadata
    )
    data_verification_report_with_survey = utils.add_survey_results(
        data_verification_report_with_metadata, survey
    )

    return data_verification_report_with_survey


if __name__ == "__main__":
    overwrite = True
    print("Preprocessing data")
    # Load the metadata spreadsheet
    metadata = utils.load_df(path=config.METADATA_PATH)

    # survey_path doesn't exist;
    if overwrite or not config.QUESTIONNAIRE_PATH.exists():
        print("Survey responses not found, loading from individual files")
        # Load the survey responses
        surveys_paths = [
            "201123_form_db_45.79.2223.150.json",
            "011023_form_db_json_from_lacclab-participant-survey.json",
            "050224_form_db_lacclab_survey.json",
        ]
        surveys = [
            utils.load_json(path=config.BASE_PATH / "raw_surveys" / survey_path)
            for survey_path in surveys_paths
        ]
        survey_responses = utils.preprocess_surveys(surveys=surveys)
        with open(file=config.QUESTIONNAIRE_PATH, mode="w") as file:
            json.dump(survey_responses, file, indent=4)
    else:
        survey_responses = utils.load_json(path=config.QUESTIONNAIRE_PATH)
    # load the trial reports
    trials = utils.load_df(path=config.TRIAL_P_PATH)
    trials = utils.values_conversion(df=trials)

    full_report = create_full_report(trials, metadata, survey_responses)
    print(f"Saving full report to {config.FULL_REPORT_PATH}")
    full_report.to_csv(config.FULL_REPORT_PATH, index=False)

    print("Filtering survey responses")
    survey_responses = utils.filter_survey_responses(survey_responses, full_report)

    print("Updating questionnaire format")
    survey_responses = utils.update_questionnaire_format(survey_responses)

    print(f"Saving questionnaire to {config.QUESTIONNAIRE_PATH}")
    with open(config.QUESTIONNAIRE_PATH, "w") as f:
        json.dump(survey_responses, f, indent=4)

    print("Processing full report to session summary")
    validation_error = pd.read_csv(config.BASE_PATH / "validation_error.csv")
    session_summary = utils.process_full_report_to_session_summary(
        full_report, validation_error
    )
    print(f"Saving session summary to {config.SESSION_SUMMARY_PATH}")
    session_summary.to_csv(config.SESSION_SUMMARY_PATH, index=False)


    subjects_from_trial_report = pd.Series(trials["RECORDING_SESSION_LABEL"].str.lower().unique())
    subjects_from_metadata = metadata["Filename"].dropna().str.lower()
    is_in_metadata = subjects_from_trial_report.isin(subjects_from_metadata)
    subjects_not_in_metadata = subjects_from_trial_report[~is_in_metadata]
    subjects_not_in_metadata.to_csv(
        config.BASE_PATH / "subjects_not_in_metadata.csv", index=False
    )
    print(f"Found {len(subjects_not_in_metadata)} subjects not in metadata, saved to {config.BASE_PATH/'subjects_not_in_metadata.csv'}")
    
    is_in_trial_report = subjects_from_metadata.isin(subjects_from_trial_report)
    subjects_not_in_trial_report = subjects_from_metadata[~is_in_trial_report]
    subjects_not_in_trial_report.to_csv(
        config.BASE_PATH / "subjects_not_in_trial_report.csv", index=False
    )

    # TODO delete if not needed
    # dat_base_path = Path('/Users/shubi/Library/CloudStorage/OneDrive-Technion/In-lab Experiments/OneStopGaze Experiment Sources/experiment-data_source/dat files')
    # dat_files_name = ['onestop_1n_l1_l60.dat', 'onestop_1p_l1_l60.dat', 'onestop_2n_l1_l60.dat', 'onestop_2p_l1_l60.dat', 'onestop_3n_l1_l60_hashtagfix.dat', 'onestop_3p_l1_l60_hashtagfix.dat']
    # new_dat_path = Path(config.BASE_PATH, 'all_dat_files_merged.tsv')

    # unpacked_columns = [
    #     "trials",
    #     "Trial_Index_",
    #     "DURATION",
    #     "BLINK_COUNT",
    #     "AVERAGE_BLINK_DURATION",
    #     "article_id",
    #     "article_ind",
    #     "level",
    #     "level_ind",
    #     "paragraph_id",
    #     "q_ind",
    #     "question",
    # ]
    # unpacked_data = DATA.explode(unpacked_columns)
