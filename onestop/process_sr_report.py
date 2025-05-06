"""This script is used to parse the raw data and save it in a format
that is easier to work with."""

import json
import logging
import os
from enum import Enum
from itertools import product
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import spacy
import torch
from tap import Tap
from text_metrics.merge_metrics_with_eye_movements import (
    add_metrics_to_word_level_eye_tracking_report,
)
from text_metrics.surprisal_extractors import extractor_switch
from tqdm import tqdm
import platform


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

    NOTE: To extract surprisal from state-spaces/mamba-* variants, better to first run:
    >>> pip install causal-conv1d
    >>> pip install mamba-ssm
    """

    SURPRISAL_MODELS: List[str] = [
        "gpt2",
    ]  # Models to extract surprisal from
    NLP_MODEL: str = "en_core_web_sm"
    parsing_mode: Literal["keep-first", "keep-all", "re-tokenize"] = "re-tokenize"

    save_path: Path = Path()  # The path to save the data.
    data_path: Path = Path()  # Path to data folder.
    onestopqa_path: Path = Path("data/onestop_qa.json")
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


def create_and_configure_logger(log_name: str = "log.log") -> logging.Logger:
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


logger = create_and_configure_logger("preprocessing.log")


def preprocess_data(args: ArgsParser) -> pd.DataFrame:
    """
    Main preprocessing pipeline for eye movement data.

    This function coordinates the entire preprocessing workflow including:
    - Data validation and loading
    - Span correction and question field fixes
    - Column renaming and standardization
    - Word span metrics computation
    - Question labeling addition
    - Word length computation
    - Paragraph extraction and processing
    - Word metrics addition for IA mode
    - Data cleaning and formatting
    - Saving processed data

    Args:
        args (ArgsParser): Configuration parameters for preprocessing

    Returns:
        pd.DataFrame: Fully preprocessed eye movement data
    """
    validate_files(args)
    df = load_data(args.data_path, sep="\t")
    df = correct_span_issues(df)
    df = fix_question_field(df)
    df = rename_columns(df)
    df = compute_word_span_metrics(df=df, mode=args.mode)
    df = add_question_labels(df, args)
    df = compute_word_length(df, args)

    if args.mode == Mode.IA and args.report == "P":
        paragraph_per_trial_extraction(df, args)
    df = add_paragraph_per_trial(df, args)

    if args.mode == Mode.IA:
        df = add_word_metrics(df, args)

    df = clean_and_format_data(df)
    print(args.save_path)
    single_value_columns = find_single_value_columns(df)
    print(single_value_columns)

    df = remove_unused_columns(df, COLUMNS_TO_DROP)

    save_processed_data(df, args)

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


def our_processing(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    LaCC lab-specific processing pipeline for OneStop dataset.

    Extends the public dataset with additional features including:
    - Integer and float feature conversions
    - Index adjustments
    - Fixation data cleaning
    - Unique paragraph ID addition
    - Word span metrics computation
    - Span-level metrics computation
    - Feature normalization
    - Question difficulty data integration
    - Previous word metrics (for IA mode)
    - Line position metrics (for IA mode)

    Args:
        df (pd.DataFrame): Input DataFrame from public preprocessing
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: Extended DataFrame with LaCC lab features
    """
    duration_field, ia_field = get_constants_by_mode(args.mode)

    df = convert_to_int_features(df, args)
    df = convert_to_float_features(df, args)
    df = adjust_indexing(df, args)
    df = drop_missing_fixation_data(df, args)
    df = add_unique_paragraph_id(df)
    df = compute_word_span_metrics(df, args.mode)
    df = compute_span_level_metrics(df, ia_field, args.mode, duration_field)
    df = compute_normalized_features(df, duration_field, ia_field)
    df = add_prolific_qas_distribution(df, args)
    df = add_reference_and_cs_two_questions(df, args)
    df = add_question_n_condition_prediction_label(df)
    df = add_lonely_and_coupled_questions_data_to_text(df, args)

    if args.mode == Mode.IA:
        df = add_previous_word_metrics(df, args)
        df = compute_start_end_line(df)
        df = add_additional_metrics(df)

    return df


def add_lonely_and_coupled_questions_data_to_text(
    df: pd.DataFrame, args: ArgsParser
) -> pd.DataFrame:
    """
    This function adds the following columns to text data:
    - "lonely_question": The question that doesn't share critical span with another.
    - "couple_question_1": The first question in a pair of questions that share a critical span.
    - "couple_question_2": The second question in a pair of questions that share a critical span.
    """

    print("Adding other questions to text data")
    text_data = df[
        [
            "article_batch",
            "article_id",
            "paragraph_id",
            "same_critical_span",
            "onestopqa_question_id",
        ]
    ].drop_duplicates()
    raw_text = get_raw_text(args)

    lonely_questions, couple_questions_1, couple_questions_2 = [], [], []
    for _, row in tqdm(
        iterable=text_data.iterrows(), total=len(text_data), desc="Adding"
    ):
        try:
            # Filter the original DataFrame for the current paragraph
            full_article_id = f"{row.article_batch}_{row.article_id}"
            questions = pd.DataFrame(
                get_article_data(article_id=full_article_id, raw_text=raw_text)[
                    "paragraphs"
                ][row.paragraph_id - 1]["qas"]
            ).drop(["answers"], axis=1)
            assert len(questions) == 3, f"Expected 3 questions for paragraph \
            {row.paragraph_id}, got {len(questions)}"

            lonely_question = questions.loc[
                questions["question_prediction_label"] == 0, "question"
            ].item()
            couple_question_1 = questions.loc[
                questions["question_prediction_label"] == 1, "question"
            ].item()
            couple_question_2 = questions.loc[
                questions["question_prediction_label"] == 2, "question"
            ].item()

            # make sure that the other questions are not the same
            assert (
                couple_question_1 != couple_question_2
            ), f"Other questions are the same: {couple_question_1}"
            # note the any of the couple questions and the lonely question can be row.question

        except ValueError:
            lonely_question = ""
            couple_question_1 = ""
            couple_question_2 = ""

        lonely_questions.append(lonely_question)
        couple_questions_1.append(couple_question_1)
        couple_questions_2.append(couple_question_2)

    # Add the other questions to the DataFrame
    text_data["couple_question_1"] = couple_questions_1
    text_data["couple_question_2"] = couple_questions_2
    text_data["lonely_question"] = lonely_questions
    df = df.merge(text_data, validate="m:1", how="left")

    return df


def convert_to_int_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to integer type.

    Handles missing values and dots by replacing them with 0 before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted integer columns
    """
    # In general, only features that have '.' or NaN or not automatically converted.

    to_int_features = [
        "article_batch",
        "article_id",
        "paragraph_id",
        "repeated_reading_trial",
        "practice_trial",
        # "question_preview",
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
    df[to_int_features] = df[to_int_features].replace({".": 0, np.nan: 0}).astype(int)
    logger.info(
        "%s fields converted to int, nan ('.') values replaced with 0.", to_int_features
    )
    return df


def convert_to_float_features(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Convert specified columns to float type.

    Handles missing values and dots by replacing them with None before conversion.
    Different columns are processed based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with converted float columns
    """
    if args.mode == Mode.IA:
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
    elif args.mode == Mode.FIXATION:
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
    df[to_float_features] = (
        df[to_float_features].replace(to_replace={".": None}).astype(float)
    )
    logger.info(
        "%s fields converted to float, nan ('.') values replaced with None.",
        to_float_features,
    )
    return df


def adjust_indexing(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Adjust indexing to be 0-indexed.

    Subtracts 1 from specified columns based on whether in IA or FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with adjusted indexing
    """
    if args.mode == Mode.IA:
        subtract_one_fields = [IA_ID_COL]
    elif args.mode == Mode.FIXATION:
        subtract_one_fields = [
            FIXATION_ID_COL,
            NEXT_FIXATION_ID_COL,
        ]
    df[subtract_one_fields] -= 1
    logger.info("%s values adjusted to be 0-indexed.", subtract_one_fields)
    return df


def drop_missing_fixation_data(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Drop rows with missing fixation data.

    Drops rows with missing values in specified columns for FIXATION mode.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Contains mode configuration

    Returns:
        pd.DataFrame: DataFrame with dropped rows
    """
    if args.mode == Mode.FIXATION:
        dropna_fields = [FIXATION_ID_COL, NEXT_FIXATION_ID_COL]
        df = df.dropna(subset=dropna_fields)
        logger.info(
            "After dropping rows with missing data in %s: %d records left in total.",
            dropna_fields,
            len(df),
        )
    return df


def add_question_n_condition_prediction_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add question and condition prediction labels.

    Adds a new column 'question_n_condition_prediction_label' based on
    'same_critical_span' and 'question_preview' values.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added labels
    """
    df["question_n_condition_prediction_label"] = df.apply(
        lambda x: x["same_critical_span"]
        if x["question_preview"] in [1, "Hunting"]
        else 3,
        axis=1,
    )  # 3 = label for null question (gathering), corresponds to cond pred.
    return df


def add_reference_and_cs_two_questions(
    df: pd.DataFrame, args: ArgsParser
) -> pd.DataFrame:
    """
    Add reference and critical span information for questions.

    Adds columns 'q_reference' and 'cs_has_two_questions' based on
    OneStopQA data.

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration parameters

    Returns:
        pd.DataFrame: DataFrame with added reference and span info
    """
    text_data = df[
        [
            "article_batch",
            "article_id",
            "paragraph_id",
            "same_critical_span",
            "onestopqa_question_id",
        ]
    ].drop_duplicates()
    cs_has_two_questions, q_references = (
        enrich_text_data_with_reference_and_cs_two_questions(text_data, args)
    )
    text_data["q_reference"] = q_references
    text_data["cs_has_two_questions"] = cs_has_two_questions
    df = df.merge(text_data, validate="m:1", how="left")
    return df


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


def add_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add additional metrics to the DataFrame.

    Adds columns for regression rate, total skip, and part length.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added metrics
    """

    logger.info("Adding additional metrics...")
    df["regression_rate"] = df["IA_REGRESSION_OUT_FULL_COUNT"] / df["IA_RUN_COUNT"]
    df["total_skip"] = df["IA_DWELL_TIME"] == 0
    df["part_length"] = df["part_max_IA_ID"] - df["part_min_IA_ID"] + 1
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


def add_unique_paragraph_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add unique paragraph ID to the DataFrame.

    Creates a new column 'unique_paragraph_id' by combining article_batch,
    article_id, difficulty_level, and paragraph_id.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        pd.DataFrame: DataFrame with added unique paragraph ID
    """
    logger.info("Adding unique paragraph id...")
    df["unique_paragraph_id"] = (
        df[["article_batch", "article_id", "difficulty_level", "paragraph_id"]]
        .astype(str)
        .apply("_".join, axis=1)
    )
    return df


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
    label_field = "IA_LABEL" if args.mode == Mode.IA else "CURRENT_FIX_LABEL"
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
            save_path.parent / name / f"{save_path.stem}_{name}.csv", index=False
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


def enrich_text_data_with_reference_and_cs_two_questions(
    text_data: pd.DataFrame, args
) -> tuple[List[int], List[str]]:
    """
    Add question reference info and critical span metadata from OneStopQA.

    For each question:
    - Gets whether critical span has multiple questions
    - Gets question reference information

    Args:
        text_data (pd.DataFrame): DataFrame with text metadata
        args: Configuration containing onestopqa_path

    Returns:
        tuple[List[int], List[str]]: Critical span flags and reference info
    """
    raw_text = get_raw_text(args)
    cs_has_two_questions = []
    q_references = []
    for row in tqdm(
        iterable=text_data.itertuples(), total=len(text_data), desc="Adding"
    ):
        full_article_id = f"{row.article_batch}_{row.article_id}"
        try:
            questions = pd.DataFrame(
                get_article_data(full_article_id, raw_text)["paragraphs"][
                    row.paragraph_id - 1  # type: ignore
                ]["qas"]
            )
            cs_two_questions_flag: int = questions.loc[
                questions["q_ind"] == row.onestopqa_question_id, "cs_has_two_questions"
            ].item()

            q_reference = questions.loc[
                questions["q_ind"] == row.onestopqa_question_id, "references"
            ].item()
        except ValueError:
            cs_two_questions_flag = 0
            q_reference = ""
        cs_has_two_questions.append(cs_two_questions_flag)
        q_references.append(q_reference)
    return cs_has_two_questions, q_references


def compute_start_end_line(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute for each word  whether it the first/last word in the line (not sentence!).

    This function adds two new columns to the input DataFrame: 'start_of_line' and 'end_of_line'.
    A word is considered to be at the start of a line if its 'IA_LEFT' value is smaller than the previous word's.
    A word is considered to be at the end of a line if its 'IA_LEFT' value is larger than the next word's.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must contain the columns 'participant_id', 'unique_paragraph_id', and 'IA_LEFT'.

    Returns:
    pd.DataFrame: The input DataFrame with two new columns: 'start_of_line' and 'end_of_line'.
    """

    logger.info("Adding start_of_line and end_of_line columns...")
    grouped_df = df.groupby(
        ["participant_id", "unique_paragraph_id", "repeated_reading_trial"]
    )
    df["start_of_line"] = (
        grouped_df["IA_LEFT"].shift(periods=1, fill_value=1000000) > df["IA_LEFT"]
    )
    df["end_of_line"] = (
        grouped_df["IA_LEFT"].shift(periods=-1, fill_value=-1) < df["IA_LEFT"]
    )
    return df


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
            "Reduced_POS": "ptb_pos",
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


def add_previous_word_metrics(df: pd.DataFrame, args: ArgsParser) -> pd.DataFrame:
    """
    Add metrics from previous words in reading sequence.

    Shifts metrics including:
    - Word frequencies
    - Word lengths
    - Surprisal values

    Args:
        df (pd.DataFrame): Input DataFrame
        args (ArgsParser): Configuration with model names

    Returns:
        pd.DataFrame: DataFrame with previous word metrics
    """
    logger.info("Calculating previous word metrics...")
    group_columns = ["participant_id", "unique_paragraph_id"]
    columns_to_shift = [
        "wordfreq_frequency",
        "subtlex_frequency",
        "word_length_no_punctuation",
    ]
    columns_to_shift += [f"{model}_surprisal" for model in args.SURPRISAL_MODELS]
    for column in columns_to_shift:
        df[f"prev_{column}"] = df.groupby(group_columns)[column].shift(1)
    return df


def compute_span_level_metrics(
    df: pd.DataFrame, ia_field: str, mode: Mode, duration_col: str
) -> pd.DataFrame:
    """
    Calculate aggregated metrics for different text spans.

    Computes:
    - Total dwell time per trial/span
    - Min/max word indices per trial/span
    - For fixations: count per span
    - Normalizes indices to start at 0

    Args:
        df (pd.DataFrame): Input DataFrame
        ia_field (str): Column name for word/fixation index
        mode (Mode): IA or FIXATION processing mode
        duration_col (str): Column name for duration values

    Returns:
        pd.DataFrame: DataFrame with added span-level metrics
    """
    logger.info("Computing span-level metrics...")

    group_by_fields = [
        "participant_id",
        "unique_paragraph_id",
        "repeated_reading_trial",
    ]

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
        df = df.merge(
            temp_max_per_trial,
            on=group_by_fields,
            validate="m:1",
            suffixes=(None, "_y"),
        )
        logger.info("Shifting IA_ID to start at 0...")
        df[ia_field] -= df["min_IA_ID"]
        df.drop(columns=["min_IA_ID", "max_IA_ID"], inplace=True)

    max_per_trial = df.groupby(group_by_fields).agg(
        total_IA_DWELL_TIME=pd.NamedAgg(column=duration_col, aggfunc="sum"),
        min_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="min"),
        max_IA_ID=pd.NamedAgg(column=ia_field, aggfunc="max"),
    )
    df = df.merge(
        max_per_trial, on=group_by_fields, validate="m:1", suffixes=(None, "_y")
    )
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
    df = df.merge(
        max_per_trial_part, on=group_by_fields, validate="m:1", suffixes=(None, "_y")
    ).copy()
    return df


def compute_normalized_features(
    df: pd.DataFrame, duration_col: str, ia_field: str
) -> pd.DataFrame:
    """
    Calculate normalized versions of key metrics.

    Adds columns for:
    - Normalized dwell times (total and by part)
    - Normalized word positions (total and by part)
    - Reverse indices from end

    Args:
        df (pd.DataFrame): Input DataFrame
        duration_col (str): Column name for duration values
        ia_field (str): Column name for word/fixation index

    Returns:
        pd.DataFrame: DataFrame with normalized metrics
    """
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

    """Validates that the spacy model is downloaded"""
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


if __name__ == "__main__":
    public_preprocess = True
    lacclab_preprocess = False
    save_path = Path("processed_reports")
    base_data_path = Path("data/Outputs")
    hf_access_token = ""  # Add your huggingface access token here
    surprisal_models = [
        # "meta-llama/Llama-2-7b-hf",
        "gpt2",
        #   "gpt2-large", "gpt2-xl",
        # "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B",
        # 'EleutherAI/gpt-j-6B',
        # "facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
        # "EleutherAI/pythia-70m",
        # "EleutherAI/pythia-160m", "EleutherAI/pythia-410m", "EleutherAI/pythia-1b",
        # "EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b", "EleutherAI/pythia-6.9b", "EleutherAI/pythia-12b",
        # "state-spaces/mamba-370m-hf", "state-spaces/mamba-790m-hf", "state-spaces/mamba-1.4b-hf", "state-spaces/mamba-2.8b-hf",
    ]

    reports = [
        "P",
        "T",
        "A",
        "QA",
        "Q_preview",
        "Q",
        "F",
    ]

    modes = [
        Mode.IA.value,
        Mode.FIXATION.value,
    ]
    short_to_long_mapping = {
        "T": "Title",
        "P": "Paragraph",
        "A": "Answers",
        "QA": "QA",
        "Q_preview": "Question_Preview",
        "Q": "Questions",
        "F": "Feedback",
    }
    for mode, report in product(modes, reports):
        if lacclab_preprocess and report not in ["P"]:
            # TODO stop skipping at some point
            print(f"Skipping {mode} report {report}")
            continue
        print(f"Processing {mode} report {report}")
        if mode == Mode.FIXATION.value:
            data_path = base_data_path / f"Fixations reports/fixations_{report}.tsv"
        else:
            data_path = base_data_path / f"raw_ia_reports/ia_{report}.tsv"
        save_file = f"{mode}_{short_to_long_mapping[report]}.csv"
        trial_level_paragraphs_path = save_path / "trial_level_paragraphs.csv"
        args = [
            "--data_path",
            str(data_path),
            "--save_path",
            str(save_path / save_file),
            "--mode",
            mode,
            "--report",
            report,
            "--trial_level_paragraphs_path",
            str(trial_level_paragraphs_path),
            "--SURPRISAL_MODELS",
            *surprisal_models,
            "--hf_access_token",
            hf_access_token,
            "--device",
            get_device(),
        ]
        cfg = ArgsParser().parse_args(args)
        if public_preprocess:
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Running preprocessing with args: {args}")
            preprocess_data(cfg)

        if lacclab_preprocess:
            df = load_data(save_path / "full" / save_file)
            df = our_processing(df=df, args=cfg)

            logger.info(
                f"Saved processed data to lacclab_processed_reports/full/{save_file}"
            )
            Path("lacclab_processed_reports/full").mkdir(parents=True, exist_ok=True)
            df.to_csv(
                Path("lacclab_processed_reports") / "full" / save_file, index=False
            )
