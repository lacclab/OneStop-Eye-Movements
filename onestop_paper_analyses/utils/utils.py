import json

import pandas as pd
from loguru import logger

logger.add(sink="preprocessing.log", level="INFO")


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


def count_sentences(paragraph):
    """
    Count the number of sentences in a given paragraph.

    Args:
        paragraph (str): The text to analyze

    Returns:
        int: Number of sentences found

    Note: Considers '.', '!', and '?' as sentence endings.
    """
    if not isinstance(paragraph, str):
        raise TypeError("Input must be a string")

    if not paragraph.strip():
        return 0

    # Split on common sentence endings
    sentence_endings = [
        ". ",
        "! ",
        "? ",
        ".\n",
        "!\n",
        "?\n",
        ".” ",
        "!” ",
        "?” ",
        ".”\n",
        "!”\n",
        "?”\n",
    ]
    count = 0

    # Handle the case where the paragraph ends without space
    if paragraph.strip()[-1] in ".!?”":
        count = 1

    for ending in sentence_endings:
        count += paragraph.count(ending)

    return count


def analyze_text_series(series):
    """
    Analyze a pandas series containing text data.

    Args:
        series (pd.Series): Series containing text data

    Returns:
        pd.DataFrame: DataFrame with text analysis metrics
    """
    # Count sentences for each row
    sentence_counts = series.apply(count_sentences)

    # Count words for each row
    word_counts = series.str.split().str.len()

    # Calculate average sentence length (words per sentence)
    avg_sentence_length = word_counts / sentence_counts

    # Create results DataFrame
    results = pd.DataFrame(
        {
            "text": series,
            "sentence_count": sentence_counts,
            "word_count": word_counts,
            "avg_words_per_sentence": avg_sentence_length.round(2),
        }
    )

    return results
