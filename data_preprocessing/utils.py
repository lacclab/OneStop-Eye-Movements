import json
import os
import platform
from enum import Enum
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import spacy
import torch
from loguru import logger
from tap import Tap
from text_metrics.merge_metrics_with_eye_movements import (
    add_metrics_to_word_level_eye_tracking_report,
)
from text_metrics.surprisal_extractors import extractor_switch
from tqdm import tqdm

from data_preprocessing import config

logger.add(sink="preprocessing.log", level="INFO")

SHORT_TO_LONG_MAPPING = {
    "T": "Title",
    "P": "Paragraph",
    "A": "Answers",
    "QA": "QA",
    "Q_preview": "Question_Preview",
    "Q": "Questions",
    "F": "Feedback",
}


class Mode(Enum):
    IA = "ia"
    FIXATION = "fixations"


IA_ID_COL = "IA_ID"
FIXATION_ID_COL = "CURRENT_FIX_INTEREST_AREA_INDEX"
NEXT_FIXATION_ID_COL = "NEXT_FIX_INTEREST_AREA_INDEX"

NUMBER_TO_LETTER_MAP = {"0": "A", "1": "B", "2": "C", "3": "D"}
COLUMNS_TO_DROP = [
    "Head_Direction",
    "AbsDistance2Head",
    "Token_idx",
    "TAG",
    "Token",
    "Word_idx",
    "IA_LABEL_y",
    "aspan_ind_start",
    "aspan_ind_end",
    "is_in_aspan",
    "dspan_ind_start",
    "dspan_ind_end",
    "is_in_dspan",
    "is_before_aspan",
    "is_after_aspan",
    "relative_to_aspan",
    "Trial_Index",
    "Trial_Index_",
    # "q_ind",
    "principle_list",
    "level_ind",
    "condition_symb",
    "a_key",
    "b_key",
    "c_key",
    "d_key",
    "batch_condition",
    "Session_Name_",
    "DATA_FILE",
    "Trial_Recycled_",
    "LETTER_HIGHT",
    "LETTER_WIDTH",
    "DUMMY",
    "COMPREHENSION_PERCENT",
    "COMPREHENSION_SCORE",
    "TRIGGER_PADDING_X",
    "TRIGGER_PADDING_Y",
    "RECALIBRATE",
    "ALL_ANSWERS",
    "ANSWER",
    "GROUPING_VARIABLES",
    "IA_DYNAMIC",
    "IA_END_TIME",
    "IA_GROUP",
    "IA_INSTANCES_COUNT",
    "IA_POINTS",
    "IA_START_TIME",
    "IA_TYPE",
    "IP_END_EVENT_MATCHED",
    "IP_INDEX",
    "IP_LABEL",
    "IP_START_EVENT_MATCHED",
    "REPORTING_METHOD",
    "TIME_SCALE",
    "is_content_word",
    "text_onscreen_version",
    "text_spacing_version",
]


class ArgsParser(Tap):
    """Args parser for preprocessing.py

        Note, for fixation data, the X_IA_DWELL_TIME, for X in
        [total, min, max, part_total, part_min, part_max]
        columns are computed based on the CURRENT_FIX_DURATION column.

        Note, documentation was generated automatically. Please check the source code for more info.
    Args:
        log_name (str): The name of the log file
        SURPRISAL_MODELS (List[str]): Models to extract surprisal from
        save_path (Path): The path to save the data
        unique_item_columns (List[str]): columns that make up a unique item
        add_prolific_qas_distribution (bool): whether to add question difficulty data from prolific
        qas_prolific_distribution_path (Path | None): Path to question difficulty data from prolific
        mode (Mode): whether to use interest area or fixation data
    """

    SURPRISAL_MODELS: List[str] = [
        "gpt2",
    ]  # Models to extract surprisal from
    NLP_MODEL: str = "en_core_web_sm"
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"] = "re-tokenize"

    save_path: Path = Path()  # The path to save the data.
    data_path: Path = Path()  # Path to data folder.
    ia_data_path: Path = Path()  # Path to ia data folder.
    onestopqa_path: Path = Path("data_preprocessing/onestop_qa.json")
    trial_level_paragraphs_path: Path = Path()

    add_prolific_qas_distribution: bool = (
        False  # whether to add question difficulty data from prolific
    )
    qas_prolific_distribution_path: Path = (
        Path()
    )  # Path to question difficulty data from prolific
    mode: Mode = Mode.IA  # whether to use interest area or fixation data
    report: str = "P"  # The report to process
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Which device to run the surprisal models on

    """ Some models supported by this function require a huggingface access token
        e.g meta-llama/Llama-2-7b-hf. If you have one, please add it here.
        https://huggingface.co/docs/hub/security-tokens
        """
    hf_access_token: str | None = None

    def process_args(self) -> None:
        validate_spacy_model(self.NLP_MODEL)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)


def load_df(path):
    str_path = str(path)
    end = ["P.tsv", "Q.tsv", "A.tsv", "QA.tsv", "preview.tsv", "T.tsv"]
    if any(str_path.endswith(suffix) for suffix in end):
        try:
            df = pd.read_table(path, encoding="utf-16", engine="pyarrow")
        except:
            df = pd.read_table(path, engine="pyarrow")
    elif str_path.endswith(".csv"):
        df = pd.read_csv(path, engine="pyarrow")
    else:
        df = pd.read_table(path, engine="pyarrow")
    return df


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def get_average_length(list_of_sentences):
    length_of_sentences = []
    for sentence in list_of_sentences:
        words = len(sentence.split())
        length_of_sentences.append(words)
    average = sum(length_of_sentences) / len(length_of_sentences)
    return average


def get_number_of_tokens(df, column):
    tokens = []
    for i in df[column]:
        tokens.append(i.split())
    return tokens


def duration_report(df):
    columns = [
        "RECORDING_SESSION_LABEL",
        "trials",
        "Trial_Index_",
        "session_duration",
        "total_duration",
        "IP_DURATION",
        "START_TIME",
        "END_TIME",
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
    duration_report = pd.DataFrame(columns=columns)

    # find out unique participant
    unique_participant = np.unique(df["RECORDING_SESSION_LABEL"].values)

    # iterate over each trial
    for i, unique_part in enumerate(unique_participant):
        # from each unique participant get the 'block' from the original dataframe
        block = df.loc[df["RECORDING_SESSION_LABEL"] == unique_part]

        new_row = block.loc[block.index.values[0]].copy()
        start_time = block["START_TIME"].min() / 1000 / 60
        end_time = block["END_TIME"].max() / 1000 / 60
        session_duration = block["DURATION"].sum() / 1000 / 60

        # mini_block = block[(block['reread'] == "0") & (block['practice'] == "0") & (block["RECALIBRATE"] == "False")]
        mini_block = block[block["RECALIBRATE"] == "False"]
        new_row["trials"] = mini_block["trial"].values.tolist()
        new_row["Trial_Index_"] = mini_block["Trial_Index_"].values.tolist()
        new_row["IP_DURATION"] = (
            sum(mini_block["IP_DURATION"].values.tolist()) / 1000 / 60
        )
        new_row["START_TIME"] = start_time
        new_row["END_TIME"] = end_time
        new_row["session_duration"] = session_duration
        new_row["total_duration"] = end_time - start_time
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

        duration_report.loc[duration_report.shape[0]] = new_row

    return duration_report


def update_json_keys(data):
    updated_data = {}

    # Direct mappings
    direct_mappings = {
        "subject_id": "Participant ID",
        # "name.first": "First name",
        # "name.last": "Last name",
        # "email": "Email",
        "age": "Age",
        "gender": "Gender",
        "home_country": "Home Country",
        "education": "Education Level",
        "affiliated": "University Affiliation",
        "native_speaker": "Native English Speaker",
        "english.startLearning": "English AoA",
    }

    for old_key, new_key in direct_mappings.items():
        if "." in old_key:
            parts = old_key.split(".")
            if parts[0] in data and parts[1] in data[parts[0]]:
                updated_data[new_key] = data[parts[0]][parts[1]]
        elif old_key in data:
            updated_data[new_key] = data[old_key]

    # Education years
    if "years_education" in data:
        updated_data["Years in Secondary/High School"] = None
        updated_data["Years in Undergraduate"] = None
        updated_data["Years in Postgraduate"] = None

        # Update based on available data
        if len(data["years_education"]) > 0:
            updated_data["Years in Secondary/High School"] = data["years_education"][
                0
            ].get("years")
        if len(data["years_education"]) > 1:
            updated_data["Years in Undergraduate"] = data["years_education"][1].get(
                "years"
            )
        if len(data["years_education"]) > 2:
            updated_data["Years in Postgraduate"] = data["years_education"][2].get(
                "years"
            )

    # University details
    if data.get("affiliated") == "yes":
        updated_data["University Institution"] = data.get("institution")
        updated_data["University Role"] = data.get("role")

    # Countries lived in
    if "countries" in data:
        updated_data["Countries Lived In"] = [
            {
                "country": country["country"],
                "fromTime": {
                    "year": country["fromTime"]["year"],
                    "month": country["fromTime"]["month"],
                },
                "toTime": {
                    "year": country["toTime"]["year"],
                    "month": country["toTime"]["month"],
                },
            }
            for country in data["countries"]
        ]

    # Reading habits
    reading_categories = [
        "Textbooks",
        "Academic",
        "Magazines",
        "Newspapers",
        "Email",
        "Fiction",
        "Nonfiction",
        "Internet",
        "Other",
    ]

    if "english" in data and "reading_frequency" in data["english"]:
        updated_data["Reading habits in English"] = {}
        for category in reading_categories:
            key = category.lower()
            if key in data["english"]["reading_frequency"]:
                updated_data["Reading habits in English"][category] = data["english"][
                    "reading_frequency"
                ][key]

    # Update to handle the new data format
    if data.get("multilingual") == "yes" and "other_languages" in data:
        for lang_data in data["other_languages"]:
            language_info = {
                "Language": lang_data.get("language"),
                "Language Proficiency": lang_data.get("proficiency"),
                "Speaking Proficiency": lang_data.get("proficiency_speaking"),
                "Understanding Proficiency": lang_data.get("proficiency_understanding"),
                "Reading Proficiency": lang_data.get("proficiency_reading"),
                "Language AoA": lang_data.get("startLearning"),
                "Language Learning Duration": lang_data.get("usedLanguage"),
            }

            # Add reading habits for native languages
            if (
                lang_data.get("proficiency") == "native"
                and "reading_frequency" in lang_data
            ):
                reading_habits = {}
                for category in reading_categories:
                    key = category.lower()
                    if key in lang_data["reading_frequency"]:
                        value = lang_data["reading_frequency"][key]
                        reading_habits[f"{category}"] = value

                language_info["Reading habits"] = reading_habits

            updated_data.setdefault("Languages", []).append(language_info)

    # Dyslexia and Language Impairments
    if "impairment" in data:
        impairment_type = data["impairment"].get("type")
        impairment_details = data["impairment"].get("details", "")

        if impairment_type == "dyslexia":
            updated_data["Dyslexia"] = "Yes"
            updated_data["Dyslexia Details"] = (
                impairment_details if impairment_details else "N/A"
            )
            updated_data["Language Impairments"] = "No"
            updated_data["Language Impairment Details"] = "N/A"
        elif impairment_type == "language_impairment":
            updated_data["Dyslexia"] = "No"
            updated_data["Dyslexia Details"] = "N/A"
            updated_data["Language Impairments"] = "Yes"
            updated_data["Language Impairment Details"] = (
                impairment_details if impairment_details else "N/A"
            )
        else:
            updated_data["Dyslexia"] = "No"
            updated_data["Language Impairments"] = "No"
    else:
        updated_data["Dyslexia"] = "No"
        updated_data["Language Impairments"] = "No"

    if "vision_impairment" in data and data["vision_impairment"].get("impairments"):
        updated_data["Eye Conditions"] = "Yes"
        updated_data["Eye Condition Details"] = data["vision_impairment"]["impairments"]
    else:
        updated_data["Eye Conditions"] = "No"
        updated_data["Eye Condition Details"] = "N/A"

    return updated_data


def change_participant_id_to_session_label(survey_responses, df):
    """
    Changes the participant_id in the survey responses to match the session label in the full report.
    """
    for record in survey_responses:
        subject_id = record.get("subject_id", None)
        if subject_id and (subject_id in df["ID"].values):
            # Get the corresponding session label
            session_label = df.loc[
                df["ID"] == subject_id, "RECORDING_SESSION_LABEL"
            ].values[0]
            print(f"Changing subject_id {subject_id} to session label {session_label}")
            record["subject_id"] = session_label
    return survey_responses


def update_questionnaire_format(data) -> list[dict] | dict:
    if isinstance(data, list):
        updated_data = [update_json_keys(item) for item in data]
    else:
        updated_data = update_json_keys(data)

    return updated_data


def filter_survey_responses(survey_responses, full_report):
    """
    Filters survey responses to keep only those with subject_ids present in the full report.

    Parameters:
    - survey_responses (list of dict): The list of survey responses, each containing a 'subject_id'.
    - full_report (pd.DataFrame): The full report DataFrame containing an 'ID' column.

    Returns:
    - list of dict: Filtered list of survey responses.
    """
    # Keep only relevant survey subject_ids
    filtered_survey_responses = []
    for record in survey_responses:
        subject_id = record.get("subject_id", None)
        if subject_id and (subject_id in full_report["ID"].values):
            filtered_survey_responses.append(record)
    return filtered_survey_responses


def process_full_report_to_session_summary(data, validation_error):
    data = data.copy()
    validation_error["file_name"] = (
        validation_error["file_name"]
        .str.replace(".asc", "", case=False, regex=False)
        .str.lower()
    )
    data = data.merge(
        validation_error,
        left_on="RECORDING_SESSION_LABEL",
        right_on="file_name",
        how="left",
    )
    data["Subject ID"] = data["RECORDING_SESSION_LABEL"]
    data = data.rename(
        columns={
            "Subject ID": "participant_id",
            "batch": "article_batch",
            "list": "list_number",
            "batch_condition": "question_preview",
            "Data Collection Site": "data_collection_site",
            "comprehension_score_without_reread": "comprehension_score-regular_trials",
            "comprehension_score_reread": "comprehension_score-repeated_reading",
            "session_interruptions_(recalibrations)": "recalibration_count",
            "total_recalibrations": "total_recalibrations",
            "avg_avg_val_error": "mean_validation_error",
            "session_duration": "session_duration",
            "total_duration": "total_session_duration",
            "dominant eye": "dominant_eye",
            "EYE_TRACKED": "tracked_eye",
            "LEXTALE": "lextale_score",
        }
    )

    # Select relevant columns
    data = data[
        [
            "participant_id",
            "article_batch",
            "list_number",
            "question_preview",
            "data_collection_site",
            "comprehension_score-regular_trials",
            "comprehension_score-repeated_reading",
            "recalibration_count",
            "total_recalibrations",
            "mean_validation_error",
            "total_session_duration",
            "session_duration",
            "dominant_eye",
            "tracked_eye",
            "lextale_score",
        ]
    ]
    return data


def correct_survey_ids(survey, mapping_path):
    print("Correcting subject_ids")
    # TODO make sure that the changes are actually saved
    # Load last_name_mapping from a file
    try:
        with open(mapping_path, "r") as file:
            last_name_mapping = json.load(file)

    except FileNotFoundError:
        print("last_name_mapping.json not found")
        last_name_mapping = {}

    for record in survey:
        last_name = record["name"].get("last", "").title()  # Convert to title case
        # get first name
        first_name = record["name"].get("first", "").title()
        # Check if the last name is in the mapping dictionary
        if last_name in last_name_mapping:
            print(
                f"Changing subject_id for {first_name} {last_name} to {last_name_mapping[last_name]}"
            )
            record["subject_id"] = last_name_mapping[last_name]
    return survey


def values_conversion(df):
    df["RECORDING_SESSION_LABEL"] = df["RECORDING_SESSION_LABEL"].map(
        lambda s: s.lower()
    )
    df["EYE_TRACKED"] = df["EYE_TRACKED"].map(lambda s: s.upper())
    df["is_correct"] = df["correct_answer"] == df["ANSWER"]
    df["RECALIBRATE"] = df["RECALIBRATE"].map(
        {
            "True": True,
            "False": False,
            "TRUE": True,
            "FALSE": False,
            True: True,
            False: False,
        }
    )
    df["practice"] = df["practice"].map({"0": 0, 0: 0, "1": 1, 1: 1})
    df["reread"] = df["reread"].map({"0": 0, 0: 0, "1": 1, 1: 1})
    for column in ["batch", "AVERAGE_BLINK_DURATION"]:
        df[column] = df[column].replace(".", 0)
    for c in [
        "batch",
        "batch_condition",
        "list",
        "paragraph_id",
        "level_ind",
        "q_ind",
        "BLINK_COUNT",
        "DURATION",
        "trial",
    ]:
        try:
            df[c] = df[c].astype(int)
        except:
            df[c] = df[c].astype(str)
    df["AVERAGE_BLINK_DURATION"] = df["AVERAGE_BLINK_DURATION"].astype(float)
    df["batch_condition"] = df["batch_condition"].map({"p": True, "n": False})
    return df


def compare_participants_metadata_trial(metadata, trials):
    participants = set(metadata["Filename"].values)
    sessions = set(np.unique(trials["RECORDING_SESSION_LABEL"].values))
    if participants == sessions:
        print("All participants have an experiment session")
    else:
        p_without_s = participants.difference(sessions)
        s_without_metadata = sessions.difference(participants)
        if p_without_s:
            print(
                f"the eye-tracking data for participants {p_without_s} are missing from the trial report"
            )
        if s_without_metadata:
            print(f"participants {s_without_metadata} are missing from the metadata")


def check_list_balance(df):
    df = df.copy()
    df = df[(df["RECALIBRATE"] == False) & (df["practice"] == 0)]
    batch = df.groupby(["batch"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    batch_condition = (
        df.groupby(["batch_condition"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    )
    level = df.groupby(["level"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    q_ind = df.groupby(["q_ind"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    list = df.groupby(["list"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    article_id = (
        df.groupby(["article_id"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    )
    paragraph_id = (
        df.groupby(["paragraph_id"])["RECORDING_SESSION_LABEL"].unique().to_dict()
    )
    count = [batch, batch_condition, list]
    c = []
    for attribute in count:
        total = 0
        par = []
        for k in attribute:
            p = len(attribute[k])
            par.append(p)
            total = total + p
        par.append(total)
        c.append(par)
    return c


def get_comprehension_score(df):
    df = df.copy()
    df = df[(df["RECALIBRATE"] == False) & (df["practice"] == 0)]
    res = df.groupby(["RECORDING_SESSION_LABEL", "reread"]).agg(
        num_correct_trials=("is_correct", "sum"),
        num_trials=("is_correct", "count"),
    )
    res["comprehension_score"] = (
        (res["num_correct_trials"] / res["num_trials"] * 100).round().astype(int)
    )
    res = res.reset_index()
    res = res.pivot(
        index="RECORDING_SESSION_LABEL", columns="reread", values="comprehension_score"
    )
    res.columns = ["comprehension_score_without_reread", "comprehension_score_reread"]
    return res


def add_comprehension_score_to_df(df, trials_scores):
    merged = trials_scores.merge(
        df,
        how="right",
        left_on="RECORDING_SESSION_LABEL",
        right_on="RECORDING_SESSION_LABEL",
        validate="1:1",
    )
    return merged


def check_if_participant_filled_survey(df):
    not_filled_survey = df.loc[df["age"] == -1]
    return not_filled_survey


def add_data_collection_site(df):
    technion_experimenter = ["Aya", "Liz", "Nethanella"]
    df["Data Collection Site"] = None
    for index, row in df.iterrows():
        if row["Experimenter"] in technion_experimenter:
            df.at[index, "Data Collection Site"] = "Technion"
        else:
            df.at[index, "Data Collection Site"] = "MIT"


def add_metadata(df, metadata):
    columns = [
        "Filename",
        "ID",
        "dominant eye",
        # "Start Time",
        "Data Collection Site",
        # "MIT/Technion",
        # "Experimenter",
        "LEXTALE",
        # "Experiment Notes",
        # "Survey notes",
    ]
    metadata_to_add = pd.DataFrame(columns=columns)
    metadata_to_add["Filename"] = metadata["Filename"]
    # metadata_to_add["MIT/Technion"] = metadata["MIT/Technion"].map({"Yes": True, "No": False})
    metadata_to_add["LEXTALE"] = metadata["LEXTALE"]
    # metadata_to_add["Experiment Notes"] = metadata["Experiment Notes"]
    # metadata_to_add["Survey notes"] = metadata["Survey notes"]
    # metadata_to_add["Experimenter"] = metadata["Experimenter"]
    metadata_to_add["ID"] = metadata["ID"].astype(str)
    metadata_to_add["dominant eye"] = metadata["dominant eye"].astype(str)
    metadata_to_add["Start Time"] = metadata["Start Time"].astype(str)
    add_data_collection_site(metadata)
    metadata_to_add["Data Collection Site"] = metadata["Data Collection Site"]
    merged_df = df.merge(
        metadata_to_add,
        how="outer",
        left_on="RECORDING_SESSION_LABEL",
        right_on="Filename",
    )
    merged_df = merged_df.drop(columns=["Filename"])
    return merged_df


def add_survey_results(df, survey):
    df_id = df["ID"].astype(str).values
    df = df.assign(age=-1)
    df = df.assign(gender=None)
    df = df.assign(native_speaker=None)
    df = df.assign(home_country=None)
    df = df.assign(english_speaking_country=None)
    df = df.assign(stat_learning_english=None)
    df = df.assign(years_in_english_country=None)
    df = df.assign(percentage_in_en_country_out_of_life=None)
    df = df.assign(additional_languages=None)
    df = df.assign(balanced_bilinguals=False)
    df = df.assign(education=None)
    df = df.assign(affiliated=None)
    df = df.assign(institution=None)
    df = df.assign(multilingual=None)
    df = df.assign(impairment=None)
    df = df.assign(vision_impairment=None)

    for subject_data in survey:
        subject_id = subject_data.get("subject_id", None)
        if subject_id and (subject_id in df_id):
            i = df.index[df["ID"] == subject_id]
            df.loc[i, "age"] = int(subject_data["age"])
            df.loc[i, "gender"] = subject_data["gender"]
            df.loc[i, "native_speaker"] = subject_data["native_speaker"]
            df.loc[i, "home_country"] = subject_data["home_country"]
            if subject_data["multilingual"] == "yes":
                additional_languages = []
                for l in subject_data["other_languages"]:
                    additional_languages.append(l["language"])
                    if l["proficiency"] == "native":
                        if "Hebrew" in l["language"]:
                            balanced_bilinguals = "Hebrew"
                        else:
                            balanced_bilinguals = l["language"]
                        df.loc[i, "balanced_bilinguals"] = balanced_bilinguals
                s_additional_languages = ""
                for l in additional_languages:
                    s_additional_languages = s_additional_languages + l + " "
                df.loc[i, "additional_languages"] = s_additional_languages
            df.loc[i, "stat_learning_english"] = subject_data["english"][
                "startLearning"
            ]
            subject_countries = ""
            years_in_country = 0
            for country in subject_data["countries"]:
                if country["country"] in config.ENGLISH_SPEAKING_COUNTRIES:
                    years_in_country = years_in_country + (
                        country["toTime"]["year"] - country["fromTime"]["year"]
                    )
                    if country["country"] not in subject_countries:
                        subject_countries = subject_countries + country["country"] + "-"
            df.loc[i, "english_speaking_country"] = subject_countries[:-1]
            df.loc[i, "years_in_english_country"] = years_in_country
            df.loc[i, "percentage_in_en_country_out_of_life"] = (
                years_in_country / subject_data["age"] * 100
            )
            df.loc[i, "education"] = subject_data["education"]
            try:
                if subject_data["affiliated"] == "yes":
                    df.loc[i, "affiliated"] = True
                    df.loc[i, "institution"] = subject_data["institution"]
                else:
                    df.loc[i, "affiliated"] = False
            except:
                print(f"{subject_id} didn't fill the affiliated q in thr survey")
            if subject_data["multilingual"] == "yes":
                df.loc[i, "multilingual"] = True
            else:
                df.loc[i, "multilingual"] = False
            if subject_data["impairment"]["type"] != "none":
                df.loc[i, "impairment"] = (
                    subject_data["impairment"]["type"]
                    + " "
                    + subject_data["impairment"]["details"]
                )
            else:
                df.loc[i, "impairment"] = subject_data["impairment"]["type"]
            try:
                df.loc[i, "vision_impairment"] = subject_data["vision_impairment"][
                    "impairments"
                ]
            except:
                pass
    return df


def validate_anonymity(survey):
    print("Make sure that the survey is anonymous")
    filtered_survey = []

    for participant in survey:
        for col in ["name", "email"]:
            if col in participant:
                del participant[col]
                print(
                    f"Removed {col} from participant {participant.get('subject_id', 'unknown')}. "
                    "There should be no identifying information in the survey."
                )

        filtered_survey.append(participant)

    return filtered_survey


def preprocess_surveys(surveys):
    # concat to one list all the surveys (each is a list of dictionaries)
    survey_responses = [
        response["subject_form"] for survey in surveys for response in survey
    ]
    survey_responses = correct_survey_ids(
        survey=survey_responses,
        mapping_path=config.BASE_PATH / "last_name_mappings.json",
    )
    survey_responses = validate_anonymity(survey=survey_responses)

    return survey_responses


def merge_dat_files(dat_base_path, dat_files_name, new_dat_path):
    dat_files = []
    for dat in dat_files_name:
        path = Path(dat_base_path, dat)
        dat_files.append(path)
    dats = []
    for dat_file in dat_files:
        print(dat_file)
        df = pd.read_csv(dat_file, sep="\t")
        df = df.iloc[1:]
        dats.append(df)
    df = pd.concat(dats)
    # remove '$' from column names
    df.columns = [col.replace("$", "") for col in df.columns]
    # delete the first row
    df.to_csv(new_dat_path, index=False, sep="\t")


def add_is_correct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_correct column to the DataFrame.
    """
    df["is_correct"] = df.selected_answer == "A"
    assert df.is_correct.nunique() == 2, "is_correct should be binary"
    return df


def add_word_metrics_fixation(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add word metrics for fixation data.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added word metrics
    """
    ia_data = load_data(args.ia_data_path)
    ia_data = ia_data.rename(
        columns={
            "IA_ID": FIXATION_ID_COL,
            "IA_LABEL": "CURRENT_FIX_INTEREST_AREA_LABEL",
        }
    )
    merge_keys = [
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
        "participant_id",
        "repeated_reading_trial",
        FIXATION_ID_COL,
        "TRIAL_INDEX",
        "CURRENT_FIX_INTEREST_AREA_LABEL",
    ]
    features = [
        "word_length_no_punctuation",
        "subtlex_frequency",
        "wordfreq_frequency",
        "gpt2_surprisal",
        "universal_pos",
        "ptb_pos",
        "head_word_index",
        "dependency_relation",
        "left_dependents_count",
        "right_dependents_count",
        "distance_to_head",
        "morphological_features",
        "entity_type",
    ]
    df = df.merge(
        ia_data[merge_keys + features].drop_duplicates(),
        on=merge_keys,
        how="left",
        validate="m:1",
    )
    return df


def find_single_value_columns(df):
    """
    Identifies columns that have only one unique value in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze

    Returns:
        pandas.Series: Series containing column names and their single unique value
    """
    # Dictionary to store columns with single values
    single_value_cols = {}

    # Check each column
    for column in df.columns:
        try:
            unique_values = df[column].nunique(dropna=False)
            if unique_values == 1:
                # Get the single unique value
                single_value = df[column].iloc[0]
                single_value_cols[column] = single_value
        except TypeError:
            # Handle unhashable columns
            try:
                unique_values = len(set(tuple(row) for row in df[column]))
                if unique_values == 1:
                    single_value = df[column].iloc[0]
                    single_value_cols[column] = single_value
            except Exception:
                continue
    # Convert to Series for better display
    result = pd.Series(single_value_cols)

    # Add count information
    print(
        f"Found {len(result)} columns with only one unique value out of {len(df.columns)} total columns"
    )

    return result


def add_question_labels(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add question labels from OneStopQA data.

    Adds a new column 'same_critical_span' based on OneStopQA data.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added question labels
    """
    text_data = df[
        ["article_batch", "article_id", "paragraph_id", "onestopqa_question_id"]
    ].drop_duplicates()
    question_prediction_labels = enrich_text_data_with_question_label(text_data, args)
    text_data["same_critical_span"] = question_prediction_labels
    df = df.merge(text_data, validate="m:1", how="left")
    return df


def add_prolific_qas_distribution(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add question difficulty data from Prolific.

    Merges question difficulty data with the main DataFrame if specified.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added question difficulty data
    """
    if args.add_prolific_qas_distribution:
        logger.info("Adding question difficulty data...")
        question_difficulty = pd.read_csv(args.qas_prolific_distribution_path)
        df = df.merge(
            question_difficulty,
            on=["article_batch", "article_id", "paragraph_id", "same_critical_span"],
            validate="m:1",
            how="left",
        )
    else:
        logger.warning(
            "Warning add_prolific_qas_distribution=%s. Not adding question difficulty data.",
            args.add_prolific_qas_distribution,
        )
    return df


def get_constants_by_mode(mode: Mode) -> tuple[str, str]:
    """
    Get constants based on processing mode.

    Returns duration and IA field names based on whether in IA or FIXATION mode.

    Args:
        mode (Mode): Processing mode (IA or FIXATION)

    Returns:
        tuple[str, str]: Duration and IA field names
    """
    duration_field = "IA_DWELL_TIME" if mode == Mode.IA else "CURRENT_FIX_DURATION"
    ia_field = IA_ID_COL if mode == Mode.IA else FIXATION_ID_COL

    return duration_field, ia_field


def validate_files(config: ArgsParser) -> None:
    """
    Validate input files exist and are accessible.

    Ensures save path and trial level paragraphs path directories exist.
    Checks if onestopqa_path and qas_prolific_distribution_path (if specified) are valid files.

    Args:
        config (ArgsParser): Configuration parameters

    Raises:
        FileNotFoundError: If onestopqa_path or qas_prolific_distribution_path not found
    """
    config.save_path.parent.mkdir(parents=True, exist_ok=True)
    config.trial_level_paragraphs_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.onestopqa_path.is_file():
        raise FileNotFoundError(
            f"No onestopqa text data found at {config.onestopqa_path}."
        )
    if config.add_prolific_qas_distribution:
        qas_prolific_distribution_path = config.qas_prolific_distribution_path
        assert os.path.exists(
            qas_prolific_distribution_path
        ), f"No question difficulty data found at {qas_prolific_distribution_path}"


def clean_and_format_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format the DataFrame.

    Standardizes column names, converts columns to appropriate types,
    and processes answers.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: Cleaned and formatted DataFrame
    """
    logger.info("Cleaning and formatting data...")
    df.columns = df.columns.str.replace(" ", "_")

    # Convert columns to appropriate types
    df = df.astype(
        {
            "question_preview": bool,
            "practice_trial": bool,
            "repeated_reading_trial": bool,
        }
    )

    # Process answers
    df["answers_order"] = df["answers_order"].apply(
        lambda x: [NUMBER_TO_LETTER_MAP[i] for i in str(x).strip("[]").split()]
    )
    df = df.copy()
    df["selected_answer"] = df.apply(
        lambda x: x["answers_order"][x["selected_answer_position"]], axis=1
    )

    df["paragraph"] = (
        df["paragraph"]
        .str.replace(r'culture" .', r'culture".')
        .str.replace(r'culture" .', r'culture".')
    )

    return df


def save_processed_data(df: pd.DataFrame, config: ArgsParser) -> None:
    """
    Save processed data to specified locations.

    Saves the full DataFrame and splits the dataset into sub-corpora based on reading conditions.

    Args:
        df (pd.DataFrame): Input DataFrame
        config (ArgsParser): Configuration parameters
    """
    logger.info("Saving processed data...")
    full_path = config.save_path.parent / "full"
    full_path.mkdir(parents=True, exist_ok=True)

    output_path = full_path / (config.save_path.stem + config.save_path.suffix)
    df.to_csv(output_path, index=False)
    if config.report == "P":
        split_save_sub_corpora(df, config.save_path)
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Data saved to {config.save_path}")


def remove_unused_columns(df: pd.DataFrame, to_drop: List[str]) -> pd.DataFrame:
    """
    Remove unused columns from the DataFrame.

    Drops specified columns from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        to_drop (List[str]): List of columns to drop

    Returns:
        pd.DataFrame: DataFrame with dropped columns
    """
    df = df[[col for col in df.columns if col not in to_drop]]
    return df


def paragraph_per_trial_extraction(df: pd.DataFrame, args: ArgsParser) -> None:
    """
    Extract paragraphs per trial and save to file.

    Groups by unique_paragraph_id and participant_id to recreate paragraph column.
    Saves the extracted paragraphs to trial_level_paragraphs_path.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters
    """
    logger.info(
        "Recreating paragraph column by grouping by unique_paragraph_id and participant_id..."
    )
    df["paragraph"] = df.groupby(
        [
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "participant_id",
            "repeated_reading_trial",
        ]
    )["IA_LABEL"].transform(lambda x: " ".join(x))

    text_onscreen_version = process_sequence_data(
        df, "IA_LEFT", output_name="text_onscreen_version"
    )
    text_spacing_version = process_sequence_data(
        df, "IA_LABEL", output_name="text_spacing_version"
    )

    df = df.merge(
        text_onscreen_version,
        on=[
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "participant_id",
        ],
        how="left",
    )
    df = df.merge(
        text_spacing_version,
        on=[
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "participant_id",
        ],
        how="left",
    )

    assert (
        df[
            [
                "participant_id",
                "article_batch",
                "article_id",
                "paragraph_id",
                "difficulty_level",
                "paragraph",
            ]
        ]
        .drop_duplicates()
        .drop(columns=["paragraph"])
        .equals(
            df[
                [
                    "participant_id",
                    "article_batch",
                    "article_id",
                    "paragraph_id",
                    "difficulty_level",
                ]
            ].drop_duplicates()
        )
    )
    paragraph_df = df[
        [
            "participant_id",
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "paragraph",
            "text_onscreen_version",
            "text_spacing_version",
        ]
    ].drop_duplicates()
    paragraph_df.to_csv(
        args.trial_level_paragraphs_path,
        index=False,
    )
    logger.info(
        "Saved paragraphs to %s",
        args.trial_level_paragraphs_path,
    )


def add_paragraph_per_trial(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add paragraph per trial to the DataFrame.

    Loads trial level paragraphs and merges with the main DataFrame to replace paragraph values.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added paragraphs
    """
    # Load trial level paragraphs
    trial_level_paragraphs = pd.read_csv(args.trial_level_paragraphs_path)

    # Merge with the main dataframe to replace paragraph values
    df = df.drop(columns=["paragraph"])
    df = df.merge(
        trial_level_paragraphs,
        on=[
            "participant_id",
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
        ],
        how="left",
        validate="m:1",
    )
    logger.info("Replaced paragraph values with trial level paragraphs from IA report.")
    return df


def compute_word_length(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Compute word length for each word.

    Adds a new column 'word_length' based on the length of the word label.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added word length
    """
    label_field = (
        "IA_LABEL" if args.mode == Mode.IA else "CURRENT_FIX_INTEREST_AREA_LABEL"
    )
    df["word_length"] = df[label_field].str.len()
    return df


def split_save_sub_corpora(df: pd.DataFrame, save_path: Path) -> None:
    """
    Split the dataset into sub-corpora based on reading conditions and save them separately.

    Creates four sub-datasets:
    - information_seeking_repeated: Repeated reading trials with question preview
    - repeated: Repeated reading trials without question preview
    - information_seeking: First reading trials with question preview
    - ordinary: First reading trials without question preview

    Args:
        df (pd.DataFrame): Input DataFrame containing full dataset
        save_path (Path): Base path where sub-corpora will be saved
    """
    # Create sub dataframes based on reread and preview conditions
    # Create boolean masks
    repeated_reading_trials = df["repeated_reading_trial"] == True  # noqa: E712
    question_preview = df["question_preview"] == True  # noqa: E712

    # Create filtered dataframes using masks
    filtered_dfs = {
        "information_seeking_repeated": df[repeated_reading_trials & question_preview],
        "repeated": df[repeated_reading_trials & ~question_preview],
        "information_seeking": df[~repeated_reading_trials & question_preview],
        "ordinary": df[~repeated_reading_trials & ~question_preview],
    }

    # Save dataframes
    for name, filtered_df in filtered_dfs.items():
        # make dir
        (save_path.parent / name).mkdir(parents=True, exist_ok=True)
        filtered_df.to_csv(
            save_path.parent / name / f"{save_path.stem}_{name}.csv.zip", index=False
        )


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names across the dataset.

    Maps original column names to standardized versions, including:
    - Experiment variables (e.g., list -> list_number)
    - Trial variables (e.g., batch -> article_batch)
    - Linguistic annotations (e.g., POS -> universal_pos)
    - STARC annotations (e.g., span_type -> auxiliary_span_type)
    - Surprisal metrics (*_Surprisal -> 8_surprisal)

    Args:
        df (pd.DataFrame): Input DataFrame with original column names

    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    renamed_columns = {
        # Experiment Variables
        "list": "list_number",
        "has_preview": "question_preview",
        "batch": "article_batch",
        "RECORDING_SESSION_LABEL": "participant_id",
        # Trial Variables
        "level": "difficulty_level",
        "trial": "trial_index",
        "practice": "practice_trial",
        "reread": "repeated_reading_trial",
        "article_ind": "article_index",
        "q_ind": "onestopqa_question_id",
        "correct_answer": "correct_answer_position",
        "FINAL_ANSWER": "selected_answer_position",
        "a": "answer_1",
        "b": "answer_2",
        "c": "answer_3",
        "d": "answer_4",
        # STARC
        "aspan_inds": "critical_span_indices",
        "dspan_inds": "distractor_span_indices",
    }

    df = df.rename(columns=renamed_columns)
    return df


def get_raw_text(args):
    """
    Load raw text data from OneStopQA JSON file.

    Args:
        args: Configuration containing onestopqa_path

    Returns:
        dict: Raw text data from OneStopQA JSON
    """
    with open(
        file=args.onestopqa_path,
        mode="r",
        encoding="utf-8",
    ) as f:
        raw_text = json.load(f)
    return raw_text["data"]


def get_article_data(article_id: str, raw_text) -> dict:
    """
    Retrieve article data from raw text by article ID.

    Args:
        article_id (str): Article identifier to look up
        raw_text (dict): Raw text data containing articles

    Returns:
        dict: Article data if found

    Raises:
        ValueError: If article ID not found
    """
    for article in raw_text:
        if article["article_id"] == article_id:
            return article
    raise ValueError(f"Article id {article_id} not found")


def enrich_text_data_with_question_label(text_data: pd.DataFrame, args) -> List[int]:
    """
    Add question prediction labels from OneStopQA to dataset.

    Matches questions based on article_batch, article_id, paragraph_id and
    onestopqa_question_id to get the corresponding question_prediction_label.

    Args:
        text_data (pd.DataFrame): DataFrame with text metadata
        args: Configuration containing onestopqa_path

    Returns:
        List[int]: Question prediction labels for each row
    """
    raw_text = get_raw_text(args)
    question_prediction_labels = []
    for row in tqdm(
        iterable=text_data.itertuples(),
        total=len(text_data),
        desc="Adding question labels",
    ):
        full_article_id = f"{row.article_batch}_{row.article_id}"
        try:
            questions = pd.DataFrame(
                get_article_data(full_article_id, raw_text)["paragraphs"][
                    row.paragraph_id - 1  # type: ignore
                ]["qas"]
            )
            question_prediction_label = questions.loc[
                questions["q_ind"] == row.onestopqa_question_id,
                "question_prediction_label",
            ].item()
        except ValueError:
            question_prediction_label = 0
        question_prediction_labels.append(question_prediction_label)
    return question_prediction_labels


def compute_word_span_metrics(
    df: pd.DataFrame,
    mode: Mode,
) -> pd.DataFrame:
    """
    Calculate word-level metrics relative to critical and distractor spans.

    Adds columns for:
    - Whether word is in critical/distractor span
    - Whether word is before/after critical span
    - For fixation mode: span info for next fixation

    Args:
        df (pd.DataFrame): Input DataFrame
        mode (Mode): IA or FIXATION processing mode

    Returns:
        pd.DataFrame: DataFrame with added span metrics
    """
    _, ia_field = get_constants_by_mode(mode)

    df[ia_field] = df[ia_field].replace({".": 0, np.nan: 0}).astype(int)
    pattern = r"\((\d+), ?(\d+)\)"  # Regex pattern to extract all span indices
    logger.info("Determining whether word is in the answer (critical) span...")
    cs_field_name = "critical_span_indices"
    ds_field_name = "distractor_span_indices"

    # Extract all critical span indices
    aspan_indices = (
        df[cs_field_name]
        .str.findall(pattern)
        .apply(lambda spans: [(int(start), int(end)) for start, end in spans])
    )
    df["is_in_aspan"] = df.apply(
        lambda row: any(
            (row[ia_field] >= start) & (row[ia_field] < end)
            for start, end in aspan_indices[row.name]
        ),
        axis=1,
    )

    logger.info("Determining whether word is in the distractor span...")
    # Extract all distractor span indices
    dspan_indices = (
        df[ds_field_name]
        .str.findall(pattern)
        .apply(lambda spans: [(int(start), int(end)) for start, end in spans])
    )
    df["is_in_dspan"] = df.apply(
        lambda row: any(
            (row[ia_field] >= start) & (row[ia_field] < end)
            for start, end in dspan_indices[row.name]
        ),
        axis=1,
    )

    logger.info(
        "Determining whether word is in the critical span, distractor span, or neither (other)..."
    )
    df["auxiliary_span_type"] = "outside"
    df.loc[df["is_in_dspan"], "auxiliary_span_type"] = "distractor"
    df.loc[df["is_in_aspan"], "auxiliary_span_type"] = "critical"
    logger.info("Span types determined.")

    assert df.query(
        "is_in_aspan == True & is_in_dspan == True"
    ).empty, "Should not be in both spans!"
    logger.info("Checked for overlapping a and d spans.")
    # TODO these only consider first cs and are dropped anyways (in public data)
    logger.info(
        "Determining whether word is in the critical span, before the span, or after the span..."
    )
    df[["aspan_ind_start", "aspan_ind_end"]] = (
        df[cs_field_name].str.extract(pattern, expand=True).astype(int)
    )
    df["is_in_aspan"] = (df[ia_field] >= df["aspan_ind_start"]) & (
        df[ia_field] < df["aspan_ind_end"]
    )

    df[["dspan_ind_start", "dspan_ind_end"]] = (
        df[ds_field_name].str.extract(pattern, expand=True).astype(int)
    )
    df["is_in_dspan"] = (df[ia_field] >= df["dspan_ind_start"]) & (
        df[ia_field] < df["dspan_ind_end"]
    )

    df["is_before_aspan"] = df[ia_field] < df["aspan_ind_start"]
    df["is_after_aspan"] = df[ia_field] >= df["aspan_ind_end"]
    df.loc[df["is_in_aspan"], "relative_to_aspan"] = "In Critical Span"
    df.loc[df["is_before_aspan"], "relative_to_aspan"] = "Before Critical Span"
    df.loc[df["is_after_aspan"], "relative_to_aspan"] = "After Critical Span"
    assert (
        df[["is_in_aspan", "is_before_aspan", "is_after_aspan"]].sum(axis=1) == 1
    ).all(), "should be exactly one of options"

    try:
        if mode == Mode.FIXATION:
            # Determine which span the next fixation falls into
            df["next_is_in_aspan"] = (
                df[NEXT_FIXATION_ID_COL] >= df["aspan_ind_start"]
            ) & (df[NEXT_FIXATION_ID_COL] < df["aspan_ind_end"])
            df["next_is_before_aspan"] = (
                df[NEXT_FIXATION_ID_COL] < df["aspan_ind_start"]
            )
            df["next_is_after_aspan"] = df[NEXT_FIXATION_ID_COL] >= df["aspan_ind_end"]
            df.loc[df["next_is_in_aspan"], "next_relative_to_aspan"] = (
                "In Critical Span"
            )
            df.loc[df["next_is_before_aspan"], "next_relative_to_aspan"] = (
                "Before Critical Span"
            )
            df.loc[df["next_is_after_aspan"], "next_relative_to_aspan"] = (
                "After Critical Span"
            )
            assert (
                df[
                    ["next_is_in_aspan", "next_is_before_aspan", "next_is_after_aspan"]
                ].sum(axis=1)
                == 1
            ).all(), "should be exactly one of options"
    except TypeError:
        logger.warning("Next fixation data has '.', skipping next fixation span info.")

    logger.info("Relative positions to the critical span determined.")
    return df


def correct_span_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known issues with span annotations in specific articles.

    Corrects critical and distractor span indices for:
    - Japan work culture article paragraph 3
    - Love hormone article paragraph 6

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with corrected span annotations
    """
    logger.info("Correcting span issues...")
    df.loc[
        (df["article_title"] == "Japan Calls Time on Long Hours Work Culture")
        & (df["paragraph_id"] == 3)
        & (df["level"] == "Adv")
        & (df["q_ind"] == 2),
        ["dspan_inds"],
    ] = "[(79, 102)]"

    df.loc[
        (df["article_title"] == "Japan Calls Time on Long Hours Work Culture")
        & (df["paragraph_id"] == 3)
        & (df["level"] == "Ele")
        & (df["q_ind"] == 2),
        ["dspan_inds"],
    ] = "[(64, 80)]"

    df.loc[
        (df["article_title"] == "Love Hormone Helps Autistic Children Bond with Others")
        & (df["paragraph_id"] == 6)
        & (df["level"] == "Adv")
        & (df["q_ind"] == 1),
        "aspan_inds",
    ] = "[(49, 67)]"
    return df


def fix_question_field(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix inconsistencies in question texts.

    For specified article/paragraph/question combinations,
    selects longer version when multiple question texts exist.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with consistent question texts
    """
    queries_to_take_long_question = [
        "batch==1 & article_id==1 & paragraph_id==7 & q_ind==2",
        "batch==1 & article_id==2 & paragraph_id==6 & q_ind==1",
        "batch==1 & article_id==8 & paragraph_id==2 & q_ind==1",
        "batch==1 & article_id==9 & paragraph_id==4 & q_ind==1",
        "batch==1 & article_id==9 & paragraph_id==5 & q_ind==0",
        "batch==3 & article_id==2 & paragraph_id==2 & q_ind==1",
        "batch==3 & article_id==3 & paragraph_id==2 & q_ind==2",
    ]

    for query in queries_to_take_long_question:
        questions = df.query(query).question.drop_duplicates().tolist()
        if len(questions) == 2:
            longer_question = max(questions, key=len)
            df.loc[df.query(query).index, "question"] = longer_question

    return df


def process_sequence_data(
    df: pd.DataFrame,
    group_col: str,
    output_name: str,
    filter_condition: str = "repeated_reading_trial==False",
) -> pd.DataFrame:
    """
    Create text version mappings by grouping similar paragraph presentations.

    Groups identical presentations based on word positions/labels to:
    - Identify distinct text versions
    - Map participants to text versions

    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str): Column to group by (IA_LEFT or IA_LABEL)
        filter_condition (str): Query to filter data

    Returns:
        pd.DataFrame: Text version mappings for each participant
    """
    # Get sequence data
    sequence = (
        df.query(filter_condition)
        .groupby(
            [
                "participant_id",
                "article_batch",
                "article_id",
                "paragraph_id",
                "difficulty_level",
            ]
        )[group_col]
        .apply(tuple)
        .reset_index()
    )

    # Group and create lists of participants
    result = (
        sequence.groupby(
            [
                "article_batch",
                "article_id",
                "paragraph_id",
                "difficulty_level",
                group_col,
            ]
        )["participant_id"]
        .apply(list)
        .reset_index()
    )

    # Add text version numbering
    result = result.assign(paragraph_length=result[group_col].apply(len))
    result = result.sort_values(by="paragraph_length", ascending=True)
    result[output_name] = result.groupby(
        [
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
        ]
    ).cumcount()
    result = result.drop(columns=["paragraph_length"])

    # Explode participant lists to rows
    result = result.explode("participant_id").reset_index(drop=True)
    result = result.drop(group_col, axis=1)

    return result


def add_word_metrics(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add linguistic metrics for each word.

    Calculates:
    - Surprisal from language models
    - Word frequency metrics
    - Word length metrics

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration for metrics calculation

    Returns:
        pd.DataFrame: DataFrame with added word metrics
    """
    logger.info("Adding surprisal, frequency, and word length metrics...")
    textual_item_key_cols = [
        "article_batch",
        "article_id",
        "paragraph_id",
        "difficulty_level",
        "text_onscreen_version",
    ]
    df["IA_ID"] -= 1
    df = add_metrics_to_word_level_eye_tracking_report(
        eye_tracking_data=df,
        surprisal_extraction_model_names=args.SURPRISAL_MODELS,
        spacy_model_name=args.NLP_MODEL,
        parsing_mode=args.parsing_mode,
        model_target_device=args.device,
        hf_access_token=args.hf_access_token,
        textual_item_key_cols=textual_item_key_cols,
        # CAT_CTX_LEFT: Buggy version from "How to Compute the Probability of a Word" (Pimentel and Meister, 2024). For the correct version, use the SurpExtractorType.PIMENTEL_CTX_LEFT
        surp_extractor_type=extractor_switch.SurpExtractorType.CAT_CTX_LEFT,
    )
    df["IA_ID"] += 1
    surprisal_cols = {
        col: col.replace("_Surprisal", "_surprisal")
        for col in df.columns
        if col.endswith("_Surprisal")
    }
    df = df.rename(
        columns=surprisal_cols
        | {
            "IA_LABEL_x": "IA_LABEL",
            # Linguistic Annotations - Big Three
            "Length": "word_length_no_punctuation",
            "Wordfreq_Frequency": "wordfreq_frequency",
            "subtlex_Frequency": "subtlex_frequency",
            # Linguistic Annotations - UD
            "POS": "universal_pos",
            "TAG": "ptb_pos",
            "Head_word_idx": "head_word_index",
            "Relationship": "dependency_relation",
            "n_Lefts": "left_dependents_count",
            "n_Rights": "right_dependents_count",
            "Distance2Head": "distance_to_head",
            "Morph": "morphological_features",
            "Entity": "entity_type",
            "Is_Content_Word": "is_content_word",
        }
    )

    return df


def participant_id_to_lower(df: pd.DataFrame) -> pd.DataFrame:
    """Convert participant_id to lowercase."""
    df["participant_id"] = df["participant_id"].str.lower()
    return df


def load_data(
    data_path: Path, usecols: list[str] | None = None, **kwargs
) -> pd.DataFrame:
    """Load data from a CSV file with automatic encoding detection.

    This function attempts to read a CSV file using different combinations of encodings
    (utf-16 and default) and engines (pyarrow and default pandas engine) to handle
    various file formats.

    Args:
        data_path (Path): Path to the CSV file to read
        usecols (list[str] | None, optional): List of columns to read.
            If None, reads all columns. Defaults to None.
        **kwargs: Additional keyword arguments passed to pandas.read_csv()

    Returns:
        pd.DataFrame: DataFrame containing the loaded CSV data

    Raises:
        ValueError: If the loaded DataFrame is empty
        UnicodeError: If file encoding cannot be determined
        ValueError: If file cannot be parsed as CSV
    """
    format_used = ""
    try:
        data = pd.read_csv(
            data_path,
            encoding="utf-16",
            engine="pyarrow",
            usecols=usecols,
            **kwargs,
        )
        format_used = "pyarrow with utf-16"
    except UnicodeError:
        data = pd.read_csv(data_path, engine="pyarrow", usecols=usecols, **kwargs)
        format_used = "pyarrow"
    except ValueError:
        try:
            data = pd.read_csv(data_path, encoding="utf-16", usecols=usecols, **kwargs)
            format_used = "default engine with utf-16"
        except UnicodeError:
            data = pd.read_csv(data_path, usecols=usecols, **kwargs)
            format_used = "default engine"

    print(f"Loaded {len(data)} rows from {data_path} using {format_used}.")

    if data.empty:
        raise ValueError(f"Error: No data found in {data_path}.")

    return data


def validate_spacy_model(spacy_model_name: str) -> None:
    """
    Validate that the specified spaCy model is downloaded.

    Checks if the spaCy model is a recognized package and is available for use.

    Args:
        spacy_model_name (str): Name of the spaCy model to validate

    Raises:
        ValueError: If the spaCy model is not recognized or not found
    """
    if spacy_model_name not in [
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
    ]:
        raise ValueError(
            f"Warning: {spacy_model_name} is not a recognized model. \
            Please use one of the specified models."
        )

    # Validates that the spacy model is downloaded
    if spacy.util.is_package(spacy_model_name):
        print(f"Using {spacy_model_name} as spacy model...")
    else:
        raise ValueError(
            f"Error: Spacy model {spacy_model_name} not found. \
            Please download the model using 'python -m spacy download {spacy_model_name}'."
        )


def get_device() -> str:
    """
    Get the appropriate device for running the surprisal models.

    Determines whether to use CUDA, MPI, or CPU based on availability.

    Returns:
        str: Device to use ('cuda', 'mpi', or 'cpu')
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif platform.system() == "Darwin":
        device = "mpi"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cpu":
        print(
            "Warning: Running on CPU. Extracting surprisal will take a long time. Consider running on GPU."
        )
    return device
