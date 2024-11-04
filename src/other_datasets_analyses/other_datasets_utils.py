import pandas as pd
import re
import numpy as np

dataset_options = ["onestop", "celer", "geco", "sbsat"]
DEGREES_PER_CHAR_ONESTOP = 0.34
DEGREES_PER_CHAR_SBSAT = 1

rounding_dict = {
    "CURRENT_FIX_DURATION": 1,
    "IA_FIRST_FIXATION_DURATION": 1,
    "IA_FIRST_RUN_DWELL_TIME": 1,
    "IA_REGRESSION_OUT_FULL_COUNT": 2,
    "IA_DWELL_TIME": 1,
    "IA_FIXATION_COUNT": 2,
    "IA_SKIP": 2,
    "NEXT_SAC_AMPLITUDE": 1,
    "IA_ZERO_TF": 2,
    "IA_FIRST_PASS_GAZE_DURATION": 1,
    "PREVIOUS_SAC_AMPLITUDE": 1,
    "IS_NEXT_REGRESSION": 2,
    "IS_PREV_REGRESSION": 2,
}

participant_id_col_name_datasets = {
    "onestop": "subject_id",
    "sbsat": "RECORDING_SESSION_LABEL",
    "geco": "PP_NR",
    "celer": "list",
}

item_id_col_name_datasets = {
    "onestop": "unique_paragraph_id",
    "sbsat": "page_name",
    "geco": "PART",
    "celer": "sentenceid",
}


def get_outcome_vars(dataset: str):
    pos_outcomes = [
        "IA_FIRST_FIXATION_DURATION",
        "IA_FIRST_RUN_DWELL_TIME",
        "IA_FIRST_PASS_GAZE_DURATION",
        "IA_DWELL_TIME",
    ]

    if dataset in ["onestop", "celer"]:
        fix_rep_outcomes = ["NEXT_SAC_AMPLITUDE", "CURRENT_FIX_DURATION"]

        non_pos_outcomes = [
            "IA_REGRESSION_OUT_FULL_COUNT",
            "IA_FIXATION_COUNT",
            "IA_ZERO_TF",
            "IA_SKIP",
        ]
    elif dataset == "sbsat":
        fix_rep_outcomes = ["PREVIOUS_SAC_AMPLITUDE", "CURRENT_FIX_DURATION"]

        non_pos_outcomes = []

    elif dataset == "geco":
        fix_rep_outcomes = []
        non_pos_outcomes = ["IA_FIXATION_COUNT", "IA_ZERO_TF", "IA_SKIP"]

    return pos_outcomes, fix_rep_outcomes, non_pos_outcomes


def process_outcome_vars_columns(
    ia_rep: pd.DataFrame,
    fix_rep: pd.DataFrame,
    pos_outcomes,
    fix_rep_outcomes,
    non_pos_outcomes,
):
    # for each column in pos_outcomes + non_pos_outcomes, replace all '.' values with 0 and make numerical if possible
    for col in pos_outcomes + non_pos_outcomes:
        if col == "IA_ZERO_TF":
            continue
        try:
            ia_rep[col] = ia_rep[col].replace(".", 0)
            ia_rep[col] = pd.to_numeric(ia_rep[col], errors="coerce")
        except:  # noqa: E722
            print(f"Could not convert {col} to numeric")
    for col in fix_rep_outcomes:
        try:
            fix_rep[col] = fix_rep[col].replace(".", np.nan)
            fix_rep[col] = pd.to_numeric(fix_rep[col], errors="coerce")
        except:  # noqa: E722
            print(f"Could not convert {col} to numeric")

    ia_rep["IA_ZERO_TF"] = (ia_rep["IA_FIRST_FIXATION_DURATION"] == 0).astype(int)


def get_ia_fixated_only(ia_rep: pd.DataFrame, dataset: str):
    ia_rep_fixated_only = ia_rep.copy()[
        (ia_rep["IA_FIRST_FIXATION_DURATION"] != 0) & (ia_rep["IA_DWELL_TIME"] != 0)
    ]  # this does nothing for sbsat

    if dataset != "sbsat":
        if dataset != "geco":
            ia_rep_fixated_only["IA_REGRESSION_OUT_FULL_COUNT"] = (
                ia_rep_fixated_only["IA_REGRESSION_OUT_FULL_COUNT"]
                .replace(".", np.nan)
                .astype(float)
            )

        ia_rep_fixated_only["IA_FIRST_PASS_GAZE_DURATION"] = ia_rep_fixated_only[
            "IA_FIRST_RUN_DWELL_TIME"
        ]
        ia_rep_fixated_only.loc[
            ia_rep_fixated_only["IA_FIRST_FIX_PROGRESSIVE"] != 1,
            "IA_FIRST_PASS_GAZE_DURATION",
        ] = np.nan

    return ia_rep_fixated_only


def load_ia_fix_reports(dataset: str):
    if dataset not in dataset_options:
        raise ValueError(f"dataset must be one of {dataset_options}")
    if dataset == "onestop":
        ia_rep = pd.read_csv(
            "/data/home/shared/onestop/processed/ia_data_enriched_360_05052024.csv",
            engine="pyarrow",
        )
        fix_rep = pd.read_csv(
            "/data/home/shared/onestop/processed/fixation_data_enriched_360_05052024.csv",
            engine="pyarrow",
        )
        fix_rep["IA_LABEL"] = fix_rep["CURRENT_FIX_INTEREST_AREA_LABEL"]
        fix_rep["IA_ID"] = fix_rep["CURRENT_FIX_INDEX"]

        # keep only NEXT_SAC_AMPLITUDE != '.' and CURRENT_FIX_INTEREST_AREA_ID != '.' in fix_rep
        fix_rep = fix_rep[~fix_rep["NEXT_SAC_AMPLITUDE"].isin(["."])]

        fix_rep["NEXT_SAC_AMPLITUDE"] = fix_rep["NEXT_SAC_AMPLITUDE"] * (
            1 / DEGREES_PER_CHAR_ONESTOP
        )

    elif dataset == "sbsat":
        # this is ALREADY fixated only
        ia_rep = pd.read_csv(
            "../../data/other_datasets/sb-sat/18sat_fixfinal_ia_fixated_only.csv",
            engine="pyarrow",
        )
        fix_rep = pd.read_csv(
            "../../data/other_datasets/sb-sat/18sat_fixfinal.csv",
            engine="pyarrow",
        )
        fix_rep = fix_rep[~fix_rep["PREVIOUS_SAC_AMPLITUDE"].isin(["."])]

        fix_rep["PREVIOUS_SAC_AMPLITUDE"] = fix_rep["PREVIOUS_SAC_AMPLITUDE"] * (
            1 / DEGREES_PER_CHAR_SBSAT
        )

        # fix_rep['IS_PREV_REGRESSION'] = fix_rep['PREVIOUS_SAC_DIRECTION'].apply(lambda x: 1 if x == 'LEFT' else 0)
    elif dataset == "geco":
        ia_rep = pd.read_csv(
            "../../data/other_datasets/GECO/GECOMonolingualReadingData.csv",
            engine="pyarrow",
        )
        fix_rep = None

        # in all column names replace WORD with IA
        ia_rep.columns = [x.replace("WORD", "IA") for x in ia_rep.columns]
        ia_rep["IA_FIRST_RUN_DWELL_TIME"] = ia_rep["IA_GAZE_DURATION"]
        ia_rep["IA_DWELL_TIME"] = ia_rep["IA_TOTAL_READING_TIME"]

    elif dataset == "celer":
        ia_rep = pd.read_csv(
            "../../data/other_datasets/celer/data_v2.0/sent_ia.tsv",
            sep="\t",
        )
        fix_rep = pd.read_csv(
            "../../data/other_datasets/celer/data_v2.0/sent_fix.tsv",
            sep="\t",
        )
        metadata = pd.read_csv(
            "../../data/other_datasets/celer/data_v2.0/metadata.tsv",
            sep="\t",
        )[["List", "L1"]]
        # in metadata, turn L1 == 'English' to L1 and L2 otherwise
        metadata["L1"] = metadata["L1"].apply(
            lambda x: "L1" if x == "English" else "L2"
        )
        metadata.rename(columns={"List": "list"}, inplace=True)
        ia_rep = ia_rep.merge(metadata, on="list")
        fix_rep = fix_rep.merge(metadata, on="list")

        # keep only NEXT_SAC_AMPLITUDE != '.' and CURRENT_FIX_INTEREST_AREA_ID != '.' in fix_rep
        fix_rep = fix_rep[
            ~fix_rep["NEXT_SAC_AMPLITUDE"]
            .astype(str)
            .str.contains(r"[^0-9\.\-]", na=False)
        ]
        fix_rep = fix_rep[~fix_rep["NEXT_SAC_AMPLITUDE"].isin(["."])]
        fix_rep = fix_rep[~fix_rep["CURRENT_FIX_INTEREST_AREA_ID"].isin(["."])]

        # Assuming `celer_ia` is a pandas DataFrame with a column 'WORD'
        n_upper = sum(
            len(re.findall(r"[A-Z]", word)) for word in ia_rep["WORD_NORM"].dropna()
        )
        n_all_char = sum(len(word) for word in ia_rep["WORD_NORM"].dropna())
        n_lower = n_all_char - n_upper
        # 0.36 degrees for lowercase letter, 0.49 for uppercase
        mean_char_visual_degrees = 0.36 * (n_lower / n_all_char) + 0.49 * (
            n_upper / n_all_char
        )
        CHARS_PER_ANGLE = 1 / mean_char_visual_degrees

        fix_rep["NEXT_SAC_AMPLITUDE"] = fix_rep["NEXT_SAC_AMPLITUDE"].astype(float)
        fix_rep["NEXT_SAC_AMPLITUDE"] = fix_rep["NEXT_SAC_AMPLITUDE"] * CHARS_PER_ANGLE

        # for some reasong IA_FIRST_FIX_PROGRESSIVE has some non 0-1 values. putting nans (5,245 records)
        ia_rep.loc[
            ia_rep["IA_FIRST_FIX_PROGRESSIVE"].isin(["."]), "IA_FIRST_FIX_PROGRESSIVE"
        ] = 5  # 5 is meaning less, it will be turned to nan anyway
        ia_rep["IA_FIRST_FIX_PROGRESSIVE"] = ia_rep["IA_FIRST_FIX_PROGRESSIVE"].astype(
            float
        )
        ia_rep.loc[
            ~ia_rep["IA_FIRST_FIX_PROGRESSIVE"].isin([0, 1]), "IA_FIRST_FIX_PROGRESSIVE"
        ] = np.nan

    return ia_rep, fix_rep


def load_and_process_dataset(dataset: str):
    ia_rep, fix_rep = load_ia_fix_reports(dataset)
    pos_outcomes, fix_rep_outcomes, non_pos_outcomes = get_outcome_vars(dataset)
    process_outcome_vars_columns(
        ia_rep, fix_rep, pos_outcomes, fix_rep_outcomes, non_pos_outcomes
    )
    ia_fixated_only = get_ia_fixated_only(ia_rep, dataset)
    return ia_rep, fix_rep, ia_fixated_only
