import pandas as pd
import re
import dropbox
from pathlib import Path
from io import BytesIO


def exclude_IAs(
    df: pd.DataFrame, remove_start_end_of_line: bool = True
) -> pd.DataFrame:
    et_data_enriched = df.copy()
    # ? Remove first and last words in each paragraph
    # For every unique_paragraph_id, subject_id, reread triplet, find the maximal and minimal IA_IDs
    # and remove the records with the minimal and maximal IA_ID
    min_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .min()
        .reset_index()
    )
    max_IA_IDs = (
        et_data_enriched.groupby(["unique_paragraph_id", "subject_id", "reread"])[
            "IA_ID"
        ]
        .max()
        .reset_index()
    )

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in min_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        min_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # remove from et_data_enriched the records with ['unique_paragraph_id', 'subject_id', 'reread', 'IA_ID'] in max_IA_IDs
    et_data_enriched = et_data_enriched.merge(
        max_IA_IDs,
        on=["unique_paragraph_id", "subject_id", "reread", "IA_ID"],
        how="left",
        indicator=True,
    )
    et_data_enriched = et_data_enriched[et_data_enriched["_merge"] == "left_only"]
    et_data_enriched = et_data_enriched.drop(columns=["_merge"])

    # ? Remove words that are not all letters (contains numbers or symbols inclusind punctuation)
    et_data_enriched = et_data_enriched.loc[
        et_data_enriched["IA_LABEL"].apply(lambda x: bool(re.match("^[a-zA-Z ]*$", x)))
    ]

    if remove_start_end_of_line:
        # if 'end of line' column is in the dataframe, remove all rows where 'end of line' == 1
        if "end_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.loc[
                et_data_enriched["end_of_line"] != True  # noqa: E712
            ]

        if "start_of_line" in et_data_enriched.columns:
            et_data_enriched = et_data_enriched.loc[
                et_data_enriched["start_of_line"] != True  # noqa: E712
            ]

    return et_data_enriched


def dropbox_upload(fig, project, path):
    format = path.split(".")[-1]
    img = BytesIO()
    fig_svg = fig.to_image(format=format)
    img.write(fig_svg)

    token = Path("src/linear_mm_utils/dropbox_access_token.txt").read_text()
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=img.getvalue(),
        path=f"/Apps/Overleaf/{project}/{path}",
        mode=dropbox.files.WriteMode.overwrite,
    )
