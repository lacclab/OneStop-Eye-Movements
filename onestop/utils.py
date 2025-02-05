import json
import numpy as np
import pandas as pd


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


def normalize_eye_conditions(df, condition_column):
    # Define a mapping function for normalization
    def map_conditions(conditions):
        normalized = []
        for condition in conditions:
            condition_lower = condition.lower()
            if "astigmatism" in condition_lower:
                normalized.append("Astigmatism")
            elif "ambylopia" in condition_lower:
                normalized.append("Amblyopia")
            elif any(
                term in condition_lower
                for term in [
                    "myopia",
                    "nearsight",
                    "near-sight",
                    "near sight",
                    "short-sight",
                    "short sight",
                    "short sited",
                    "nearsidedness",
                ]
            ):
                normalized.append("Lens-Corrected Myopia")
            elif "pseudotumor cerebri" in condition_lower:
                normalized.append("Pseudotumor Cerebri (cured)")
            elif "one eye does not see as well" in condition_lower:
                normalized.append(
                    "Other: One eye does not see as well as the other, but cannot be corrected with glasses."
                )
            elif "glasses" in condition_lower and len(condition_lower.split()) == 1:
                normalized.append("Glasses")
            else:
                normalized.append(condition)
        return sorted(set(normalized))

    # Apply normalization and create exploded dataframe
    df["Normalized Condition"] = df[condition_column].apply(map_conditions)
    df[["Condition1", "Condition2"]] = df["Normalized Condition"].apply(pd.Series)

    return df
