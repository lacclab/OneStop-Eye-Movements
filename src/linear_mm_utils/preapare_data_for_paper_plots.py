import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import re
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dropbox
from pathlib import Path
from io import BytesIO
import matplotlib.pyplot as plt


def upload(fig, project, path):
    format = path.split(".")[-1]
    img = BytesIO()
    fig_svg = fig.to_image(format=format)
    img.write(fig_svg)

    token = Path(
        "/data/home/meiri.yoav/OneStopGaze-Preprocessing/reread_analysis/dropbox_access_token.txt"
    ).read_text()
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=img.getvalue(),
        path=f"/Apps/Overleaf/{project}/{path}",
        mode=dropbox.files.WriteMode.overwrite,
    )


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


root_path = os.path.dirname(os.path.dirname(os.getcwd()))
data_path = root_path + "/data"
figs_path = root_path + "/reread_analysis/figs"

print("root path: ", root_path)
print("data path: ", data_path)
print("figs path: ", figs_path)


# add the root path to sys
if root_path not in sys.path:
    sys.path.append(root_path)
    sys.path.append(os.path.dirname(os.getcwd()))
    sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))


from process_IA_rep_for_reread_analysis import exclude_IAs
from calc_means_main import plot_means
from calc_means_main import calc_mm_means_for_all_outcomes
from response_to_ling import config, Normal, run_linear_mm, Poisson, Bernoulli
import plotly.graph_objects as go
import plotly.subplots as sp

outcome_name_mapping = {
    "IA_DWELL_TIME": "Total Fixation Duration",
    "IA_FIRST_FIXATION_DURATION": "First Fixation Duration",
    "IA_FIRST_PASS_GAZE_DURATION": "First Pass Gaze Duration",
    "IA_FIRST_RUN_DWELL_TIME": "Gaze Duration",
    "IA_RUN_COUNT": "Fixation Count",
    "NEXT_SAC_AMPLITUDE": "Saccade Length",
    "PREVIOUS_SAC_AMPLITUDE": "Saccade Length",
    "IA_FIRST_PASS_GAZE_DURATION": "First Pass Gaze Duration",
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
    'IS_NEXT_REGRESSION': "Regression Rate",
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
    'IA_ZERO_TF': '',
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
    "relative_to_aspan": "Relative to Critical Span",
}

all_outcomes = ['IA_FIRST_FIXATION_DURATION',
 'IA_FIRST_RUN_DWELL_TIME',
 'IA_FIRST_PASS_GAZE_DURATION',
 'IA_RUN_COUNT',
 'IA_ZERO_TF',
 'IA_REGRESSION_OUT_FULL_COUNT']


def check_same_span_2_trials(df):
    read_is_in_span = (
        df.loc[lambda x: x["reread"] == 0]
        .sort_values(by=["IA_ID"])["relative_to_aspan"]
        .reset_index(drop=True)
    )
    reread_is_in_span = (
        df.loc[lambda x: x["reread"] == 1]
        .sort_values(by=["IA_ID"])["relative_to_aspan"]
        .reset_index(drop=True)
    )
    assert len(read_is_in_span) == len(reread_is_in_span)
    return (read_is_in_span == reread_is_in_span).all()


def get_et_data_enriched_for_reread_analysis(et_data_enriched: pd.DataFrame | None = None):
    if et_data_enriched is None:
        et_data_enriched = pd.read_csv(
            # data_path + "/interim/et_data_for_reread_analysis_all_participants.csv"
            data_path + "/interim/ia_data_enriched_360_31032024.csv"
        )
    print(len(et_data_enriched))
    et_data_enriched = et_data_enriched.sort_values(
        by=["subject_id", "article_ind", "unique_paragraph_id", "IA_ID"]
    )
    # convert all '.' values in IA_REGRESSION_OUT_FULL_COUNT to Nan
    et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"] = et_data_enriched[
        "IA_REGRESSION_OUT_FULL_COUNT"
    ].replace(".", np.nan)
    # turn the rest of the column to numeric
    et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"] = pd.to_numeric(
        et_data_enriched["IA_REGRESSION_OUT_FULL_COUNT"]
    )

    # Add the parent of the current folder to sys path
    if os.path.dirname(os.getcwd()) not in sys.path:
        sys.path.append(os.path.dirname(os.getcwd()))

    print("Excluding words with numbers, punctuation, and start-end of paragrapgh")
    exclusion_df = True
    if exclusion_df:
        et_data_enriched = exclude_IAs(et_data_enriched)
    print(len(et_data_enriched))

    et_data_enriched["first_second_reading_types"] = et_data_enriched[
        "article_ind"
    ].apply(lambda x: 0 if x < 10 else x - 9)
    # # put first_second_reading_types nan where article_ind_of_first_reading is not a number
    # if df['article_ind_of_first_reading'].isna() and df['first_read_out_of_2'] == False put nan instead of first_second_reading_types. use loc
    et_data_enriched.loc[
        (et_data_enriched["article_ind_of_first_reading"].isna())
        & (et_data_enriched["first_read_out_of_2"] == False),
        "first_second_reading_types",
    ] = np.nan

    # convert 0 to <=10, 1 to 10, 2 to 11 and 3 to 12
    et_data_enriched.loc[
        et_data_enriched["first_second_reading_types"] == 0,
        "first_second_reading_types",
    ] = " <10"
    et_data_enriched.loc[
        et_data_enriched["first_second_reading_types"] == 1,
        "first_second_reading_types",
    ] = "10"
    et_data_enriched.loc[
        et_data_enriched["first_second_reading_types"] == 2,
        "first_second_reading_types",
    ] = "11"
    et_data_enriched.loc[
        et_data_enriched["first_second_reading_types"] == 3,
        "first_second_reading_types",
    ] = "12"

    # has_preview_reread will be the pair of has_preview and has_reread concatenated by _
    et_data_enriched["has_preview_reread"] = (
        et_data_enriched["has_preview"].astype(str)
        + ", "
        + et_data_enriched["reread"].astype(str)
    )

    # create a new column article_ind_group_of_first_reading where if 1 <= article_ind_of_first_reading <= 4 gets (1,4), if 5 <= article_ind_of_first_reading <= 9 gets (5,9) and if 10 get 10
    et_data_enriched["article_ind_group_of_first_reading"] = et_data_enriched[
        "article_ind_of_first_reading"
    ].apply(
        lambda x: "(1,4)"
        if (x >= 1) & (x <= 4)
        else ("(5,9)" if (x >= 5) & (x <= 9) else ("10" if x == 10 else np.nan))
    )

    # remove all subject_id, paragraph_id pairs which have a record with Length == 0
    et_data_enriched = et_data_enriched.groupby(
        ["subject_id", "unique_paragraph_id"]
    ).filter(lambda x: (x["Length"] != 0).all())

    same_span_in_12_reread = (
        et_data_enriched.query(
            "(first_read_out_of_2 == 1 and article_ind < 10) or article_ind == 12"
        )
        .groupby(["subject_id", "unique_paragraph_id"])
        .apply(lambda x: check_same_span_2_trials(x))
    )
    same_span_in_11_reread = (
        et_data_enriched.query("article_ind == 10 or article_ind == 11")
        .groupby(["subject_id", "unique_paragraph_id"])
        .apply(lambda x: check_same_span_2_trials(x))
    )

    same_span_in_12_reread = same_span_in_12_reread.reset_index().rename(
        columns={0: "same_span_in_FR"}
    )
    same_span_in_11_reread = same_span_in_11_reread.reset_index().rename(
        columns={0: "same_span_in_FR"}
    )

    # concatenate them to one df
    same_span_in_12_reread["article_ind"] = 12
    same_span_in_11_reread["article_ind"] = 11

    same_span_in_FR = pd.concat(
        [same_span_in_12_reread, same_span_in_11_reread], axis=0
    )

    # "same_question_as_FR" will be '.' for reread == 0 and same_span_in_12_reread for reread == 1 (joined on subject_id, unique_paragraph_id). Replace nans with '.'
    et_data_enriched = et_data_enriched.merge(
        same_span_in_FR,
        on=["subject_id", "unique_paragraph_id", "article_ind"],
        how="left",
        suffixes=("", "_y"),
    )
    # replace nans with '.'
    et_data_enriched["same_span_in_FR"] = et_data_enriched["same_span_in_FR"].fillna(
        "."
    )

    # put et_data_enriched['same_span_in_FR'] == '.' were et_data_enriched['has_preview'] == Gathering
    et_data_enriched.loc[
        et_data_enriched["has_preview"] == "Gathering", "same_span_in_FR"
    ] = "."

    same_span_in_FR_also_gathering_df = same_span_in_FR.copy()
    # rename the same_span_in_FR column to same_span_in_FR_also_gathering
    same_span_in_FR_also_gathering_df = same_span_in_FR_also_gathering_df.rename(
        columns={"same_span_in_FR": "same_span_in_FR_also_gathering"}
    )
    et_data_enriched = et_data_enriched.merge(
        same_span_in_FR_also_gathering_df,
        on=["subject_id", "unique_paragraph_id", "article_ind"],
        how="left",
        suffixes=("", "_y"),
    )
    et_data_enriched.loc[
        et_data_enriched["article_ind"] < 11, "same_span_in_FR_also_gathering"
    ] = "."
    et_data_enriched['IA_ZERO_TF'] = et_data_enriched['IA_DWELL_TIME'] == 0
    print(et_data_enriched)
    return et_data_enriched

def main():
    get_et_data_enriched_for_reread_analysis()

if __name__ == "__main__":
    main()
