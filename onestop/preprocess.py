import json
from pathlib import Path

import config
import numpy as np
import pandas as pd


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


def remove_participants_identification(survey):
    print("Removing participants' identification")
    filtered_survey = []

    for participant in survey:
        if "name" in participant:
            del participant["name"]
        if "email" in participant:
            del participant["email"]

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
    survey_responses = remove_participants_identification(survey=survey_responses)

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
    data["Subject ID"] = data["RECORDING_SESSION_LABEL"].str.split("_").str[1]
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
