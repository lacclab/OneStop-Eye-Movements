"""This script is used to parse the raw data and save it in a format
that is easier to work with."""

import json
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import spacy
import torch
from tap import Tap
from text_metrics.merge_metrics_with_eye_movements import add_metrics_to_eye_tracking
from tqdm import tqdm

try:
    _ = spacy.load("en_core_web_sm")
except OSError:
    print(
        "Downloading spacy model: en_core_web_sm (python -m spacy download en_core_web_sm)"
    )
    command = "python -m spacy download en_core_web_sm"
    subprocess.run(command, shell=True)


class Mode(Enum):
    IA = "ia"
    FIXATION = "fixations"


IA_ID_COL = "IA_ID"
FIXATION_ID_COL = "CURRENT_FIX_INTEREST_AREA_INDEX"
NEXT_FIXATION_ID_COL = "NEXT_FIX_INTEREST_AREA_INDEX"


class ArgsParser(Tap):
    """Args parser for preprocessing.py

        Note, for fixation data, the X_IA_DWELL_TIME, for X in
        [total, min, max, part_total, part_min, part_max]
        columns are computed basd on the CURRENT_FIX_DURATION column.

        Note, documentation was generated automatically. Please check the source code for more info.
    Args:
        log_name (str): The name of the log file
        filter_query (str): The query to filter the data by
        SURPRISAL_MODELS (List[str]): Models to extract surprisal from
        base_cols (List[str]): Also includes surprisal models
        cols_to_add (List[str]): columns to add to base_cols
        cols_to_remove (List[str]): columns to remove from base_cols
        save_path (Path): The path to save the data
        hunting_data_path (Path): Path to hunting data. Should be 'preview_p_{self.mode}.tsv'
        gathering_data_path (Path): Path to gathering data. Should be 'nopreview_p_{self.mode}.tsv'
        unique_item_columns (List[str]): columns that make up a unique item
        unique_item_column (str): defined as unique_item_columns separated by "_"
        item_column (List[str]): column that defines an item
        subject_column (List[str]): column that defines a subject
        add_prolific_qas_distribution (bool): whether to add question difficulty data from prolific
        qas_prolific_distribution_path (Path | None): Path to question difficulty data from prolific
        add_question_in_prompt (bool): whether to add the question in the prompt
        mode (Mode): whether to use interest area or fixation data
    """

    filter_query: str = ""

    """
    NOTE: To extract surprisal from state-spaces/mamba-* variants, better to first run:
    >>> pip install causal-conv1d
    >>> pip install mamba-ssm
    """
    SURPRISAL_MODELS: List[str] = [
        "gpt2",
        # "EleutherAI/gpt-j-6B",
    ]  # Models to extract surprisal from
    NLP_MODEL: str = "en_core_web_sm"
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"] = "re-tokenize"
    base_cols: List[str] = [
        "subject_id",
        "unique_paragraph_id",
        "has_preview",
        "q_ind",
        "question_prediction_label",
        "question_n_condition_prediction_label",
        "practice",
        "reread",
        "is_correct",
        "correct_answer",
        "abcd_answer",
        "answers_order",
        "list",
        "cs_has_two_questions",
        "q_reference",
        "FINAL_ANSWER",
        IA_ID_COL,
        "IA_LABEL",
        "Wordfreq_Frequency",
        "Length",
        "prev_Length",
        "prev_Wordfreq_Frequency",
        "subtlex_Frequency",
        "prev_subtlex_Frequency",
        "PARAGRAPH_RT",
        "IA_FIRST_FIX_PROGRESSIVE",
        "is_in_dspan",
        "is_in_aspan",
        "is_before_aspan",
        "is_after_aspan",
        "relative_to_aspan",
        "span_type",
        "entropy",
        "question",
        "article_ind",
        "a_proportion",
        "b_proportion",
        "c_proportion",
        "d_proportion",
        "total_IA_DWELL_TIME",
        "min_IA_ID",
        "max_IA_ID",
        "part_total_IA_DWELL_TIME",
        "part_min_IA_ID",
        "part_max_IA_ID",
        "part_num_fixations",
        "normalized_dwell_time",
        "normalized_part_dwell_time",
        "normalized_part_ID",
        "reverse_ID",
        "reverse_part_ID",
        "part_ID",
        "normalized_ID",
        "Token",
        "POS",
        "TAG",
        "Head_word_idx",
        "Relationship",
        "n_Lefts",
        "n_Rights",
        "Distance2Head",
        "Morph",
        "Entity",
        "Head_Direction",
        "Token_idx",
        "Word_idx",
        "Is_Content_Word",
        "Reduced_POS",
        "start_of_line",
        "end_of_line",
        "IA_AVERAGE_FIX_PUPIL_SIZE",
        "IA_DWELL_TIME",
        "IA_DWELL_TIME_%",
        "IA_FIRST_RUN_LANDING_POSITION",
        "IA_LAST_RUN_LANDING_POSITION",
        "IA_FIXATION_%",
        "IA_FIXATION_COUNT",
        "IA_REGRESSION_IN_COUNT",
        "IA_REGRESSION_OUT_FULL_COUNT",
        "IA_RUN_COUNT",
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIRST_FIXATION_VISITED_IA_COUNT",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_FIRST_RUN_FIXATION_%",
        "IA_FIRST_RUN_FIXATION_COUNT",
        "IA_SKIP",
        "IA_REGRESSION_PATH_DURATION",
        "IA_REGRESSION_OUT_COUNT",
        "IA_SELECTIVE_REGRESSION_PATH_DURATION",
        "IA_SPILLOVER",
        "IA_FIRST_SACCADE_AMPLITUDE",
        "IA_FIRST_SACCADE_ANGLE",
        "IA_LAST_FIXATION_DURATION",
        "IA_LAST_RUN_DWELL_TIME",
        "IA_LAST_RUN_FIXATION_%",
        "IA_LAST_RUN_FIXATION_COUNT",
        "IA_LAST_SACCADE_AMPLITUDE",
        "IA_LAST_SACCADE_ANGLE",
        "IA_LEFT",
        "IA_TOP",
        "TRIAL_DWELL_TIME",
        "TRIAL_FIXATION_COUNT",
        "TRIAL_IA_COUNT",
        "TRIAL_INDEX",
        "TRIAL_TOTAL_VISITED_IA_COUNT",
        "regression_rate",
        "total_skip",
        "part_length",
        FIXATION_ID_COL,  # Only for fixation data
        "CURRENT_FIX_INTEREST_AREA_LABEL",  # Only for fixation data
        "CURRENT_FIX_DURATION",  # Only for fixation data
        "CURRENT_FIX_PUPIL",  # Only for fixation data
        NEXT_FIXATION_ID_COL,  # Only for fixation data
        "next_relative_to_aspan",  # Only for fixation data
        "CURRENT_FIX_X",  # Only for fixation data
        "CURRENT_FIX_Y",  # Only for fixation data
        "CURRENT_FIX_INDEX",
        "NEXT_FIX_ANGLE",
        "PREVIOUS_FIX_ANGLE",
        "NEXT_FIX_DISTANCE",
        "PREVIOUS_FIX_DISTANCE",
        "NEXT_SAC_AMPLITUDE",
        "NEXT_SAC_ANGLE",
        "NEXT_SAC_AVG_VELOCITY",
        # "NEXT_SAC_BLINK_DURATION", # Mostly nans
        "NEXT_SAC_DURATION",
        "NEXT_SAC_PEAK_VELOCITY",
        "CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE",
        "NEXT_SAC_END_X",
        "NEXT_SAC_START_X",
        "NEXT_SAC_END_Y",
        "NEXT_SAC_START_Y",
    ]  # Also includes surprisal models

    cols_to_add: List[str] = []  # columns to add to base_cols
    cols_to_remove: List[str] = []  # columns to remove from base_cols

    save_path: Path = Path()  # The path to save the data.
    hunting_data_path: Path = Path()  # Path to hunting data.
    gathering_data_path: Path = Path()  # Path to gathering data.
    data_path: Path = Path()  # Path to data folder.
    onestopqa_path: Path = Path("data/interim/onestop_qa.json")
    unique_item_columns: List[str] = [
        "batch",
        "article_id",
        "level",
        "paragraph_id",
    ]  # columns that make up a unique item
    unique_item_column: str = (
        "unique_paragraph_id"  # defined as unique_item_columns separated by "_"
    )

    # groups

    item_column: List[str] = ["article_id"]  # column that defines an item
    subject_column: List[str] = ["subject_id"]  # column that defines a subject

    add_prolific_qas_distribution: bool = (
        False  # whether to add question difficulty data from prolific
    )
    qas_prolific_distribution_path: Path | None = None
    add_question_in_prompt: bool = False  # whether to add the question in the prompt
    mode: Mode = Mode.IA  # whether to use interest area or fixation data
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Which device to run the surprisal models on

    """ Some models supported by this function require a huggingface access token
        e.g meta-llama/Llama-2-7b-hf. If you have one, please add it here.
        https://huggingface.co/docs/hub/security-tokens"""
    hf_access_token: str = None

    def process_args(self) -> None:
        validate_spacy_model(self.NLP_MODEL)

        self.base_cols = list(
            set(self.base_cols)
            .union(self.cols_to_add)
            .difference(set(self.cols_to_remove))
        )
        print(f"Using columns: {self.base_cols}")


def create_and_configer_logger(log_name: str = "log.log") -> logging.Logger:
    """
    Creates and configures a logger
    Args:
        log_name (): The name of the log file
    Returns:
    """
    # set up logging to file
    logging.basicConfig(
        filename=log_name,
        level=logging.INFO,
        format="[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s",
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logger_ = logging.getLogger(__name__)
    return logger_


logger = create_and_configer_logger("preprocessing.log")


def _load_raw_data(data_path) -> pd.DataFrame:
    """Loads the raw data"""
    data = pd.read_csv(data_path, sep="\t", low_memory=False)
    logger.info("Loaded %d records from %s.", len(data), data_path)
    return data


def preprocess_data(args: ArgsParser) -> pd.DataFrame:
    # If this won't be here the Surprisal columns won't be added to the base columns if they are not in the base columns in the first place
    args.base_cols += [
        surprisal_model + "_Surprisal" for surprisal_model in args.SURPRISAL_MODELS
    ]
    args.base_cols += [
        "prev_" + surprisal_model + "_Surprisal"
        for surprisal_model in args.SURPRISAL_MODELS
    ]

    logger.info("Making sure data paths exist...")

    if args.add_prolific_qas_distribution:
        qas_prolific_distribution_path = args.qas_prolific_distribution_path
        assert os.path.exists(
            qas_prolific_distribution_path
        ), f"No question difficulty data found at {qas_prolific_distribution_path}"
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.onestopqa_path.is_file():
        extract_cs_two_questions_flag = False
        print(
            f"No onestopqa text data found at {args.onestopqa_path}. Skipping addition of 'cs_has_two_questions'."
        )
    else:
        extract_cs_two_questions_flag = True

    logger.info("Preprocessing data...")

    if args.data_path:
        df = load_data(args.data_path, sep="\t")
    else:
        hunting_data = load_data(args.hunting_data_path, sep="\t")
        gathering_data = load_data(args.gathering_data_path, sep="\t")
        df = pd.concat([hunting_data, gathering_data]).copy()
        del hunting_data, gathering_data

    df = df.loc[
        :, ~df.columns.str.contains("FSA")
    ].copy()  # TODO Temp, delete after removing from raw data

    # In general, only features that have '.' or NaN or not automatically converted.
    to_int_features = [
        "batch",
        "article_id",
        "paragraph_id",
        "reread",
        "practice",
        "has_preview",
    ]
    if args.mode == Mode.IA:
        to_int_features += [
            "IA_DWELL_TIME",
            "IA_FIRST_FIXATION_DURATION",
            "IA_REGRESSION_PATH_DURATION",
            "IA_FIRST_RUN_DWELL_TIME",
            "IA_FIXATION_COUNT",
            "IA_REGRESSION_IN_COUNT",
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_RUN_COUNT",
            "IA_FIRST_FIXATION_VISITED_IA_COUNT",
            "IA_FIRST_RUN_FIXATION_COUNT",
            "IA_SKIP",
            "IA_REGRESSION_OUT_COUNT",
            "IA_SELECTIVE_REGRESSION_PATH_DURATION",
            "IA_SPILLOVER",
            "IA_LAST_FIXATION_DURATION",
            "IA_LAST_RUN_DWELL_TIME",
            "IA_LAST_RUN_FIXATION_COUNT",
            "IA_LEFT",
            "IA_TOP",
            "TRIAL_DWELL_TIME",
            "TRIAL_FIXATION_COUNT",
            "TRIAL_IA_COUNT",
            "TRIAL_INDEX",
            "TRIAL_TOTAL_VISITED_IA_COUNT",
            "IA_FIRST_FIX_PROGRESSIVE",
        ]

        to_float_features = [
            "IA_AVERAGE_FIX_PUPIL_SIZE",
            "IA_DWELL_TIME_%",
            "IA_FIXATION_%",
            "IA_FIRST_RUN_FIXATION_%",
            "IA_FIRST_SACCADE_AMPLITUDE",
            "IA_FIRST_SACCADE_ANGLE",
            "IA_LAST_RUN_FIXATION_%",
            "IA_LAST_SACCADE_AMPLITUDE",
            "IA_LAST_SACCADE_ANGLE",
            "IA_FIRST_RUN_LANDING_POSITION",
            "IA_LAST_RUN_LANDING_POSITION",
        ]
        subtract_one_fields = [IA_ID_COL]

    elif args.mode == Mode.FIXATION:
        to_int_features += [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            "CURRENT_FIX_DURATION",
            "CURRENT_FIX_PUPIL",
            "CURRENT_FIX_X",
            "CURRENT_FIX_Y",
            "CURRENT_FIX_INDEX",
            "NEXT_SAC_DURATION",
        ]
        to_float_features = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
            "NEXT_FIX_ANGLE",
            "PREVIOUS_FIX_ANGLE",
            "NEXT_FIX_DISTANCE",
            "PREVIOUS_FIX_DISTANCE",
            "NEXT_SAC_AMPLITUDE",
            "NEXT_SAC_ANGLE",
            "NEXT_SAC_AVG_VELOCITY",
            "NEXT_SAC_PEAK_VELOCITY",
            "NEXT_SAC_END_X",
            "NEXT_SAC_START_X",
            "NEXT_SAC_END_Y",
            "NEXT_SAC_START_Y",
        ]
        subtract_one_fields = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
        ]
    else:
        raise ValueError(f"Unknown mode {args.mode}")

    # TODO Think of this more carefully - should we replace with 0 or None?
    df[to_float_features] = (
        df[to_float_features].replace(to_replace={".": None}).astype(float)
    )
    logger.info(
        "%s fields converted to float, nan ('.') values replaced with None.",
        to_float_features,
    )

    df[to_int_features] = df[to_int_features].replace({".": 0, np.nan: 0}).astype(int)
    logger.info(
        "%s fields converted to int, nan ('.') values replaced with 0.", to_int_features
    )

    df[subtract_one_fields] -= 1
    logger.info("%s values adjusted to be 0-indexed.", subtract_one_fields)

    if args.filter_query:
        df = df.query(args.filter_query).copy()
        logger.info(
            "After query: %s \n %d records left in total.", args.filter_query, len(df)
        )
    else:
        logger.info("No query applied. %d records in total.", len(df))

    if args.mode == Mode.FIXATION:
        dropna_fields = [FIXATION_ID_COL, NEXT_FIXATION_ID_COL]
        df = df.dropna(subset=dropna_fields)
        logger.info(
            "After dropping rows with missing data in %s: %d records left in total.",
            dropna_fields,
            len(df),
        )

    df = correct_span_issues(df)

    ia_field = IA_ID_COL if args.mode == Mode.IA else FIXATION_ID_COL
    df = compute_word_span_metrics(df, args.mode, ia_field)

    logger.info("Getting whether the answer is correct and the answer letter...")
    df["is_correct"] = df["correct_answer"] == df["FINAL_ANSWER"]
    df["answers_order"] = df["answers_order"].str.strip("[]").str.split()
    df["abcd_answer"] = df.apply(
        lambda x: x["answers_order"][x["FINAL_ANSWER"]], axis=1
    )
    logger.info("Replacing numeric answer with letters...")
    df.abcd_answer.replace({"0": "A", "1": "B", "2": "C", "3": "D"}, inplace=True)

    logger.info("Replacing numeric condition with words...")
    df.has_preview.replace({0: "Gathering", 1: "Hunting"}, inplace=True)

    if args.add_prolific_qas_distribution:
        logger.info("Adding question difficulty data...")
        question_difficulty = pd.read_csv(qas_prolific_distribution_path)
        df = df.merge(
            question_difficulty,
            on=["batch", "article_id", "paragraph_id", "q_ind"],
            validate="m:1",
            how="left",
        )
    else:
        logger.warning(
            "Warning add_prolific_qas_distribution=%s. Not adding question difficulty data.",
            args.add_prolific_qas_distribution,
        )

    logger.info("Adding unique paragraph id...")
    df["unique_paragraph_id"] = (
        df[args.unique_item_columns].astype(str).apply("_".join, axis=1)
    )

    logger.info("Renaming column 'RECORDING_SESSION_LABEL' to 'subject_id'...")
    df["subject_id"] = df["RECORDING_SESSION_LABEL"]

    duration_col = "IA_DWELL_TIME" if args.mode == Mode.IA else "CURRENT_FIX_DURATION"

    df = _compute_span_level_metrics(df, ia_field, args.mode, duration_col)

    if args.mode == Mode.IA:
        df = compute_start_end_line(df)  # TODO add to fixation data as well?
        df = add_word_metrics(df, args)

        df["regression_rate"] = df["IA_REGRESSION_OUT_FULL_COUNT"] / df["IA_RUN_COUNT"]
        df["total_skip"] = df["IA_DWELL_TIME"] == 0
        df["part_length"] = df["part_max_IA_ID"] - df["part_min_IA_ID"] + 1

    df = compute_normalized_features(df, duration_col, ia_field)

    if extract_cs_two_questions_flag:
        # TODO move to a separate function?
        text_data = df[
            [
                "batch",
                "article_id",
                "paragraph_id",
                "q_ind",
            ]
        ].drop_duplicates()

        with open(
            file=args.onestopqa_path,
            mode="r",
            encoding="utf-8",
        ) as f:
            RAW_TEXT = json.load(f)

        def get_article_data(article_id: str) -> dict:
            for article in RAW_TEXT["data"]:
                if article["article_id"] == article_id:
                    return article
            raise ValueError(f"Article id {article_id} not found")

        cs_has_two_questions = []
        q_references = []
        question_prediction_labels = []
        for row in tqdm(
            iterable=text_data.itertuples(), total=len(text_data), desc="Adding"
        ):
            full_article_id = f"{row.batch}_{row.article_id}"
            questions = pd.DataFrame(
                get_article_data(full_article_id)["paragraphs"][row.paragraph_id - 1][
                    "qas"
                ]
            )

            cs_two_questions_flag: int = questions.loc[
                questions["q_ind"] == row.q_ind, "cs_has_two_questions"
            ].item()
            cs_has_two_questions.append(cs_two_questions_flag)

            q_reference = questions.loc[
                questions["q_ind"] == row.q_ind, "references"
            ].item()

            question_prediction_label = questions.loc[
                questions["q_ind"] == row.q_ind, "question_prediction_label"
            ].item()

            question_prediction_labels.append(question_prediction_label)
            q_references.append(q_reference)

        text_data["cs_has_two_questions"] = cs_has_two_questions
        text_data["q_reference"] = q_references
        text_data["question_prediction_label"] = question_prediction_labels
        df = df.merge(text_data, validate="m:1", how="left")

        # {"Gathering": 0, "Hunting": 1}
        df["question_n_condition_prediction_label"] = df.apply(
            lambda x: x["question_prediction_label"]
            if x["has_preview"] in [1, "Hunting"]
            else 3,
            axis=1,
        )  # 3 is the label for the null question (gathering) and corresponds to the condition prediction.

    df = filter_columns(df, args.base_cols)

    df.to_csv(args.save_path)

    logger.info("Total number of rows: %d", len(df))
    logger.info("Data preprocessing complete. Saved to %s", args.save_path)
    return df


def compute_start_end_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute for each word  whether it the first/last word in the line (not sentence!).

    This function adds two new columns to the input DataFrame: 'start_of_line' and 'end_of_line'.
    A word is considered to be at the start of a line if its 'IA_LEFT' value is smaller than the previous word's.
    A word is considered to be at the end of a line if its 'IA_LEFT' value is larger than the next word's.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must contain the columns 'subject_id', 'unique_paragraph_id', and 'IA_LEFT'.

    Returns:
    pd.DataFrame: The input DataFrame with two new columns: 'start_of_line' and 'end_of_line'.
    """

    logger.info("Adding start_of_line and end_of_line columns...")
    grouped_df = df.groupby(["subject_id", "unique_paragraph_id"])
    df["start_of_line"] = (
        grouped_df["IA_LEFT"].shift(periods=1, fill_value=1000000) > df["IA_LEFT"]
    )
    df["end_of_line"] = (
        grouped_df["IA_LEFT"].shift(periods=-1, fill_value=-1) < df["IA_LEFT"]
    )
    return df


def compute_word_span_metrics(
    df: pd.DataFrame, mode: Mode, ia_field: str
) -> pd.DataFrame:
    pattern = r"(\d+), ?(\d+)"  # Regex pattern to extract span indices
    logger.info("Determining whether word is in the answer (critical) span...")
    df[["aspan_ind_start", "aspan_ind_end"]] = df.aspan_inds.str.extract(
        pattern, expand=True
    ).astype(int)  # TODO only the first span is extracted
    df["is_in_aspan"] = (df[ia_field] >= df["aspan_ind_start"]) & (
        df[ia_field] < df["aspan_ind_end"]
    )

    logger.info("Determining whether word is in the distractor span...")
    df[["dspan_ind_start", "dspan_ind_end"]] = df.dspan_inds.str.extract(
        pattern, expand=True
    ).astype(int)  # TODO only the first span is extracted
    df["is_in_dspan"] = (df[ia_field] >= df["dspan_ind_start"]) & (
        df[ia_field] < df["dspan_ind_end"]
    )

    logger.info(
        "Determining whether word is in the critical span, distractor span, or neither (other)..."
    )
    df["span_type"] = "other"
    df.loc[df["is_in_dspan"], "span_type"] = "d_span"
    df.loc[df["is_in_aspan"], "span_type"] = "a_span"
    logger.info("Span types determined.")

    assert df.query(
        "is_in_aspan == True & is_in_dspan == True"
    ).empty, "Should not be in both spans!"
    logger.info("Checked for overlapping a and d spans.")

    logger.info(
        "Determining whether word is in the critical span, before the span, or after the span..."
    )
    df["is_before_aspan"] = df[ia_field] < df["aspan_ind_start"]
    df["is_after_aspan"] = df[ia_field] >= df["aspan_ind_end"]
    df.loc[df["is_in_aspan"], "relative_to_aspan"] = "In Critical Span"
    df.loc[df["is_before_aspan"], "relative_to_aspan"] = "Before Critical Span"
    df.loc[df["is_after_aspan"], "relative_to_aspan"] = "After Critical Span"
    assert (
        df[["is_in_aspan", "is_before_aspan", "is_after_aspan"]].sum(axis=1) == 1
    ).all(), "should be exactly one of options"

    if mode == Mode.FIXATION:
        # Determine which span the next fixation falls into
        df["next_is_in_aspan"] = (df[NEXT_FIXATION_ID_COL] >= df["aspan_ind_start"]) & (
            df[NEXT_FIXATION_ID_COL] < df["aspan_ind_end"]
        )
        df["next_is_before_aspan"] = df[NEXT_FIXATION_ID_COL] < df["aspan_ind_start"]
        df["next_is_after_aspan"] = df[NEXT_FIXATION_ID_COL] >= df["aspan_ind_end"]
        df.loc[df["next_is_in_aspan"], "next_relative_to_aspan"] = "In Critical Span"
        df.loc[df["next_is_before_aspan"], "next_relative_to_aspan"] = (
            "Before Critical Span"
        )
        df.loc[df["next_is_after_aspan"], "next_relative_to_aspan"] = (
            "After Critical Span"
        )
        assert (
            df[["next_is_in_aspan", "next_is_before_aspan", "next_is_after_aspan"]].sum(
                axis=1
            )
            == 1
        ).all(), "should be exactly one of options"

    logger.info("Relative positions to the critical span determined.")
    return df


def correct_span_issues(df: pd.DataFrame) -> pd.DataFrame:
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


def filter_columns(df: pd.DataFrame, base_cols: List[str]) -> pd.DataFrame:
    # log the columns that were dropped
    dropped_columns = set(df.columns).difference(base_cols)
    logger.info("Dropped columns: %s", dropped_columns)

    # log the columns that were in base_cols but not in the data
    missing_columns = set(base_cols).difference(df.columns)
    logger.info("Missing columns: %s", missing_columns)

    logger.info("Final columns: %s", df.columns)

    logger.info("Keeping selected columns...")
    df = df[df.columns.intersection(base_cols)].copy()

    return df


def add_word_metrics(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    logger.info("Adding surprisal, frequency, and word length metrics...")
    df = add_metrics_to_eye_tracking(
        eye_tracking_data=df,
        surprisal_extraction_model_names=args.SURPRISAL_MODELS,
        spacy_model_name=args.NLP_MODEL,
        add_question_in_prompt=args.add_question_in_prompt,
        parsing_mode=args.parsing_mode,
        model_target_device=args.device,
        hf_access_token=args.hf_access_token,
    )

    logger.info("Renaming column 'IA_LABEL_x' to 'IA_LABEL'...")
    df.rename(columns={"IA_LABEL_x": "IA_LABEL"}, inplace=True)

    logger.info("Calculating previous word metrics...")
    group_columns = ["subject_id", "unique_paragraph_id"]
    columns_to_shift = [
        "Wordfreq_Frequency",
        "subtlex_Frequency",
        "Length",
    ]
    columns_to_shift += [f"{model}_Surprisal" for model in args.SURPRISAL_MODELS]
    for column in columns_to_shift:
        df[f"prev_{column}"] = df.groupby(group_columns)[column].shift(1)

    return df


def _compute_span_level_metrics(
    df: pd.DataFrame, ia_field: str, mode: Mode, duration_col: str
) -> pd.DataFrame:
    logger.info("Computing span-level metrics...")

    group_by_fields = ["subject_id", "unique_paragraph_id", "reread"]

    # Fix trials where ID does not start at 0
    if mode == Mode.IA:
        temp_max_per_trial = df.groupby(group_by_fields).agg(
            min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="min"),
            max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="max"),
        )
        non_zero_min_ia_id_trials = temp_max_per_trial[
            temp_max_per_trial["min_IA_ID"] != 0
        ]
        logger.info(
            "Number of trials where min_IA_ID is not zero: %d out of %d trials.",
            len(non_zero_min_ia_id_trials),
            len(temp_max_per_trial),
        )
        df = df.merge(temp_max_per_trial, on=group_by_fields, validate="m:1")
        logger.info("Shifting IA_ID to start at 0...")
        df[ia_field] -= df["min_IA_ID"]
        df.drop(columns=["min_IA_ID", "max_IA_ID"], inplace=True)

    max_per_trial = df.groupby(group_by_fields).agg(
        total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc="sum"),
        min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="min"),
        max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="max"),
    )
    df = df.merge(max_per_trial, on=group_by_fields, validate="m:1")
    group_by_fields += ["relative_to_aspan"]

    if mode == Mode.IA:
        max_per_trial_part = df.groupby(group_by_fields).agg(
            part_total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc="sum"),
            part_min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="min"),
            part_max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="max"),
        )
    elif mode == Mode.FIXATION:
        max_per_trial_part = df.groupby(group_by_fields).agg(
            part_total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc="sum"),
            part_min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="min"),
            part_max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="max"),
            part_num_fixations=pd.NamedAgg(column=ia_field, aggfunc="count"),
        )
    df = df.merge(max_per_trial_part, on=group_by_fields, validate="m:1").copy()
    return df


def compute_normalized_features(
    df: pd.DataFrame, duration_col: str, ia_field: str
) -> pd.DataFrame:
    logger.info("Computing normalized dwell time, and normalized word indices...")
    df = df.assign(
        normalized_dwell_time=df[duration_col] / df.total_IA_DWELL_TIME,
        normalized_part_dwell_time=df[duration_col] / df.part_total_IA_DWELL_TIME,
        normalized_part_ID=(df[ia_field] - df.part_min_IA_ID)
        / (df.part_max_IA_ID - df.part_min_IA_ID),
        reverse_ID=df[ia_field] - df.max_IA_ID,
        reverse_part_ID=df[ia_field] - df.part_max_IA_ID,
        part_ID=df[ia_field] - df.part_min_IA_ID + 1,
        normalized_ID=(df[ia_field] - df.min_IA_ID) / (df.max_IA_ID - df.min_IA_ID),
    ).copy()
    return df


def load_data(
    data_path: Path, has_preview_to_numeric: bool = False, **kwargs
) -> pd.DataFrame:
    if data_path.is_dir():
        try:
            dataframes = [
                pd.read_csv(file, encoding="utf-16", **kwargs)
                for file in data_path.glob("*.tsv")
            ]
        except UnicodeError:
            dataframes = [
                pd.read_csv(file, low_memory=False, **kwargs)
                for file in data_path.glob("*.tsv")
            ]
        data = pd.concat(dataframes, ignore_index=True)
    else:
        try:
            print(f"Load data from {data_path} using pyarrow.")
            data = pd.read_csv(data_path, encoding="utf-16", engine="pyarrow", **kwargs)
        except UnicodeError:
            print(
                f"Attempting to load data from {data_path} without specifying encoding."
            )
            data = pd.read_csv(data_path, engine="pyarrow", **kwargs)
        except ValueError:
            print(f"Load data from {data_path} (without pyarrow -- much slower!).")
            try:
                data = pd.read_csv(data_path, encoding="utf-16", **kwargs)
            except UnicodeError:
                print(
                    f"Attempting to load data from {data_path} without specifying encoding."
                )
                data = pd.read_csv(data_path, **kwargs)

    if has_preview_to_numeric:
        data["has_preview"] = data["has_preview"].map({"Gathering": 0, "Hunting": 1})

    logger.info("Loaded %d records from %s.", len(data), data_path)

    return data


def validate_spacy_model(spacy_model_name: str) -> None:
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

    """Validates that the spacy model is downloaded"""
    if spacy.util.is_package(spacy_model_name):
        print(f"Using {spacy_model_name} as spacy model...")
    else:
        raise ValueError(
            f"Error: Spacy model {spacy_model_name} not found. \
            Please download the model using 'python -m spacy download {spacy_model_name}'."
        )


def process_data(args: List[str], args_file: Path, save_path: Path):
    cfg = ArgsParser().parse_args(args)

    args_save_path = save_path / args_file
    save_path.mkdir(parents=True, exist_ok=True)
    cfg.save(str(args_save_path))
    print(f"Saved config to {args_save_path}")

    print(f"Running preprocessing with args: {args}")
    preprocess_data(cfg)


if __name__ == "__main__":
    data_path = Path("/data/home/shared/onestop/")
    save_path = Path("data")
    hf_access_token = ""  # Add your huggingface access token here
    filter_query = "practice==0"
    surprisal_models = [
        # "meta-llama/Llama-2-7b-hf",
        "gpt2",
        #  "gpt2-medium", "gpt2-large", "gpt2-xl",
        # "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
        # 'EleutherAI/gpt-j-6B',
        # "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
        # "EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
        # "state-spaces/mamba-370m-hf", "state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf", "state-spaces/mamba-2.8b-hf",
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for mode in [Mode.IA.value, Mode.FIXATION.value]:
        save_file = f"{mode}_P.csv"
        args_file = Path(f"{mode}_args.json")

        args = [
            "--data_path",
            str(data_path),
            "--save_path",
            str(save_path / save_file),
            "--mode",
            mode,
            "--filter_query",
            filter_query,
            "--SURPRISAL_MODELS",
            *surprisal_models,
            "--hf_access_token",
            hf_access_token,
            "--device",
            device,
        ]
        process_data(args, args_file, save_path)
