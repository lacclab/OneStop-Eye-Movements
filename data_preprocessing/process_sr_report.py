"""This script is used to parse the raw data and save it in a format
that is easier to work with."""

from itertools import product
from pathlib import Path

import pandas as pd

from data_preprocessing import utils
from data_preprocessing.utils import ArgsParser, Mode


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
    utils.validate_files(args)
    df = utils.load_data(args.data_path, sep="\t")
    df = utils.correct_span_issues(df)
    df = utils.fix_question_field(df)
    df = utils.rename_columns(df)
    df = utils.compute_word_span_metrics(df=df, mode=args.mode)
    df = utils.add_question_labels(df, args)
    df = utils.compute_word_length(df, args)

    if args.mode == Mode.IA and args.report == "P":
        utils.paragraph_per_trial_extraction(df, args)
    df = utils.add_paragraph_per_trial(df, args)

    if args.mode == Mode.IA:
        df = utils.add_word_metrics(df, args)
    elif args.mode == Mode.FIXATION:
        df = utils.add_word_metrics_fixation(df, args)

    df = utils.clean_and_format_data(df)
    df = utils.add_is_correct(df)
    print(args.save_path)
    single_value_columns = utils.find_single_value_columns(df)
    print(single_value_columns)

    df = utils.remove_unused_columns(df, utils.COLUMNS_TO_DROP)

    df = utils.participant_id_to_lower(df)

    utils.save_processed_data(df, args)

    return df


def main():
    save_path = Path("to_osf")
    base_data_path = Path("raw_sr_reports")
    hf_access_token = ""  # Add your huggingface access token here
    surprisal_models = [
        "gpt2",
        # Additional models can be used by uncommenting the lines below
        # "meta-llama/Llama-2-7b-hf",
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

    for mode, report in product(modes, reports):
        print(f"Processing {mode} report {report}")
        if mode == Mode.FIXATION.value:
            data_path = base_data_path / f"fixation_reports/fixations_{report}.tsv"
        else:
            data_path = base_data_path / f"ia_reports/ia_{report}.tsv"

        save_file = f"{mode}_{utils.SHORT_TO_LONG_MAPPING[report]}.csv.zip"
        ia_data_path = f"ia_{utils.SHORT_TO_LONG_MAPPING[report]}.csv.zip"
        trial_level_paragraphs_path = save_path / "trial_level_paragraphs.csv"
        args = [
            "--data_path",
            str(data_path),
            "--ia_data_path",
            str(save_path / "full" / ia_data_path),
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
            utils.get_device(),
        ]
        cfg = ArgsParser().parse_args(args)

        print(f"Running preprocessing with args: {args}")
        preprocess_data(cfg)


if __name__ == "__main__":
    main()
