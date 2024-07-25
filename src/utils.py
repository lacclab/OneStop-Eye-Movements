import json

import numpy as np
import pandas as pd
from datasets import load_dataset
import config
from pathlib import Path


def load_df(path):
    str_path = str(path)
    end = ["P.tsv", "Q.tsv", "A.tsv", "QA.tsv", "preview.tsv", "T.tsv"]
    if any(str_path.endswith(suffix) for suffix in end):
        try:
            df = pd.read_table(path, encoding="utf-16")
        except:
            df = pd.read_table(path)
    elif str_path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_table(path)
    return df


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


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


def add_metadata(df, metadata):
    columns = [
        "Filename",
        "ID",
        "dominant eye",
        "Start Time",
        "MIT/Technion",
        "Experimenter",
        "LEXTALE",
        "Experiment Notes",
        "Survey notes",
    ]
    metadata_to_add = pd.DataFrame(columns=columns)
    metadata_to_add["Filename"] = metadata["Filename"]
    metadata_to_add["MIT/Technion"] = metadata["MIT/Technion"].map(
        {"Yes": True, "No": False}
    )
    metadata_to_add["LEXTALE"] = metadata["LEXTALE"].fillna(-1)
    metadata_to_add["Experiment Notes"] = metadata["Experiment Notes"]
    metadata_to_add["Survey notes"] = metadata["Survey notes"]
    metadata_to_add["Experimenter"] = metadata["Experimenter"]
    metadata_to_add["ID"] = metadata["ID"].astype(str)
    # metadata_to_add['ID'] = [id[:-2] for id in metadata_to_add['ID']]
    metadata_to_add["dominant eye"] = metadata["dominant eye"].astype(str)
    metadata_to_add["Start Time"] = metadata["Start Time"].astype(str)
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
    english_countries = [
        "United States",
        "Australia",
        "United Kingdom",
        "Canada",
        "Ireland",
        "South Africa",
        "Nigeria",
    ]

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
                if country["country"] in english_countries:
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


def get_average_length(list_of_sentences):
    length_of_sentences = []
    for sentence in list_of_sentences:
        words = len(sentence.split())
        length_of_sentences.append(words)
    average = sum(length_of_sentences) / len(length_of_sentences)
    return average


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
        "english.startLearning": "English AoA"
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
            updated_data["Years in Secondary/High School"] = data["years_education"][0].get("years")
        if len(data["years_education"]) > 1:
            updated_data["Years in Undergraduate"] = data["years_education"][1].get("years")
        if len(data["years_education"]) > 2:
            updated_data["Years in Postgraduate"] = data["years_education"][2].get("years")
    
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
                    "year": country['fromTime']['year'],
                    "month": country['fromTime']['month']
                },
                "toTime": {
                    "year": country['toTime']['year'],
                    "month": country['toTime']['month']
                }
            }
            for country in data["countries"]
        ]
        
    # Reading habits
    reading_categories = [
        "Textbooks", "Academic", "Magazines", "Newspapers", "Email",
        "Fiction", "Nonfiction", "Internet", "Other"
    ]

    if "english" in data and "reading_frequency" in data["english"]:
        updated_data["Reading habits in English"] = {}
        for category in reading_categories:
            key = category.lower()
            if key in data["english"]["reading_frequency"]:
                updated_data["Reading habits in English"][category] = data["english"]["reading_frequency"][key]

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
                "Language Learning Duration": lang_data.get("usedLanguage")
            }
            
            # Add reading habits for native languages
            if lang_data.get("proficiency") == "native" and "reading_frequency" in lang_data:
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
            updated_data["Dyslexia Details"] = impairment_details if impairment_details else "N/A"
            updated_data["Language Impairments"] = "No"
            updated_data["Language Impairment Details"] = "N/A"
        elif impairment_type == "language_impairment":
            updated_data["Dyslexia"] = "No"
            updated_data["Dyslexia Details"] = "N/A"
            updated_data["Language Impairments"] = "Yes"
            updated_data["Language Impairment Details"] = impairment_details if impairment_details else "N/A"
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

def update_questionnaire_format(input_file, output_file) -> list[dict] | dict:
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        updated_data = [update_json_keys(item) for item in data]
    else:
        updated_data = update_json_keys(data)
    
    with open(output_file, 'w') as f:
        json.dump(updated_data, f, indent=4)
    return updated_data
