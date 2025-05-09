import argparse
import os
from pathlib import Path

import requests
from tqdm import tqdm
import zipfile


"""
Based on https://github.com/DiLi-Lab/PoTeC/blob/main/download_data_files.py
"""


def download_data(
    extract: bool, download_asc: bool, download_edf: bool, output_folder: str, mode: str
) -> None:
    """
    Downloads and optionally extracts eyetracking data from the OneStop OSF repository.

    Args:
        extract (bool): Whether to extract the downloaded zip files.
        download_asc (bool): Whether to download the asc files.
        download_edf (bool): Whether to download the edf files.
        output_folder (str): The folder where the downloaded files will be saved.
        mode (str): The mode of data to download. Options are 'full', 'repeated',
                    'information-seeking', 'ordinary', 'information-seeking-in-repeated'.

    Returns:
        None
    """
    base_url = "https://osf.io/download/"

    urls = {
        "ordinary": {"fixations": "9v2at", "ia": "bhf5u"},
        "information-seeking": {"fixations": "2r3pd", "ia": "dmjs4"},
        "repeated": {"fixations": "cj8rh", "ia": "e4h9a"},
        "information-seeking-in-repeated": {"fixations": "cg67z", "ia": "eunvq"},
        "asc_files": "",
        "edf_files": "",
        "metadata": {"session_summary": "yvu5w", "questionnaire": "a765t"},
        "all-regimes": {"fixations": "6jbge", "ia": "p97e5"},
        "full": {
            "fixations_title": "hr4jv",
            "ia_title": "7nsz4",
            "fixations_question_preview": "p8mfs",
            "ia_question_preview": "5fkr7",
            "fixations_paragraph": "6jcsr",
            "ia_paragraph": "m6jwy",
            "fixations_questions": "a8xvd",
            "ia_questions": "6b9f8",
            "fixations_answers": "vc23j",
            "ia_answers": "pvfdh",
            "fixations_qa": "kv6r3",
            "ia_qa": "qd4f6",
            "fixations_feedback": "cvayd",
            "ia_feedback": "a8ftm",
        },
    }
    subsets = [
        "repeated",
        "information-seeking",
        "ordinary",
        "information-seeking-in-repeated",
        "all-regimes",
    ]

    folder = Path(__file__).parent / output_folder
    if not os.path.exists(folder):
        os.makedirs(folder)

    for data, resource in (pbar := tqdm(urls.items())):
        # Skip the data if it is not in the specified mode
        if data in subsets and (mode != data or mode == "full"):
            continue

        if (data == "asc_files" and not download_asc) or (
            data == "edf_files" and not download_edf
        ):
            continue

        pbar.set_description(
            f"Downloading {'and extracting ' if extract else ''}{data}"
        )
        for resource_name, resource in resource.items():
            # Downloading the file by sending the request to the URL
            url = base_url + resource
            print(f"Downloading {data} - {resource_name} ({url}) to {folder}")
            req = requests.get(url, stream=True)

            # create new paths for the downloaded files
            filename = f"{data}.zip"
            path = folder / filename
            extract_path = folder / data

            if os.path.exists(path):
                print(f"\nPath for {data} already exists. Not downloaded to {path}")
                continue

            if os.path.exists(extract_path):
                print(
                    f"\nPath for {data} already exists. Not downloaded to {extract_path}"
                )
                continue

            # Writing the file to the local file system
            with open(path, "wb") as output_file:
                for chunk in req.iter_content(chunk_size=128):
                    output_file.write(chunk)

            if extract:
                extract_path = folder
                try:
                    with zipfile.ZipFile(path, "r") as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"Extracted {data} to {extract_path}")
                except zipfile.BadZipFile:
                    print(f"Error: {path} is not a zip file.")
                except Exception as e:
                    print(f"Error extracting {path}: {e}")
                os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract eyetracking data from the OneStop OSF repository."
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract downloaded data.",
        default=True,
    )

    parser.add_argument(
        "--asc",
        dest="download_asc",
        action="store_true",
        help="Whether to download the asc files. Default is False.",
        default=False,
    )

    parser.add_argument(
        "--edf",
        dest="download_edf",
        action="store_true",
        help="Whether to download the edf files. Default is False.",
        default=False,
    )

    parser.add_argument(
        "-o",
        "--output-folder",
        dest="output_folder",
        type=str,
        help="Path to the output folder. Default is OneStop",
        default="OneStop",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "full",
            "all-regimes",
            "repeated",
            "information-seeking",
            "ordinary",
            "information-seeking-in-repeated",
        ],
        help="Mode of data to download. Options are full, all-regimes, repeated, information-seeking, ordinary, information-seeking-in-repeated. Default is full.",
        default="full",
    )

    args = parser.parse_args()
    download_data(
        extract=args.extract,
        download_asc=args.download_asc,
        download_edf=args.download_edf,
        output_folder=args.output_folder,
        mode=args.mode,
    )
