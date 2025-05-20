import argparse
import os
import zipfile
from pathlib import Path

import requests

# Based on https://github.com/DiLi-Lab/PoTeC/blob/main/download_data_files.py


def download_data(extract: bool, output_folder: str, mode: str) -> None:
    """
    Downloads and optionally extracts eyetracking data from the OneStop OSF repository.

    Args:
        extract (bool): Whether to extract the downloaded zip files.
        output_folder (str): The folder where the downloaded files will be saved.
        mode (str): The mode of data to download. Options are 'onestop-full', 'onestop', 'repeated',
                    'information-seeking', 'ordinary', 'information-seeking-in-repeated'. Default is 'ordinary'.

    Returns:
        None
    """
    base_url = "https://osf.io/download/"

    urls = {
        "ordinary": {"fixations": "9v2at", "ia": "bhf5u"},
        "information-seeking": {"fixations": "2r3pd", "ia": "dmjs4"},
        "repeated": {"fixations": "cj8rh", "ia": "e4h9a"},
        "information-seeking-in-repeated": {"fixations": "cg67z", "ia": "eunvq"},
        "onestop": {"fixations": "6jbge", "ia": "p97e5"},
        "onestop-full": {
            "fixations_title": "hr4jv",
            "ia_title": "7nsz4",
            "fixations_question_preview": "p8mfs",
            "ia_question_preview": "k457m",
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

    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    resources = urls.get(mode)
    if resources is None:
        raise ValueError(
            f"Invalid mode '{mode}'. Available options are: {', '.join(urls.keys())}"
        )

    for resource_name, resource in resources.items():
        # Downloading the file by sending the request to the URL
        url = base_url + resource
        print(f"Downloading {mode} - {resource_name} ({url}) to {folder}")
        req = requests.get(url, stream=True)

        # create new paths for the downloaded files
        filename = f"{mode}.zip"
        path = folder / filename
        extract_path = folder / mode

        if os.path.exists(path):
            print(f"\nPath for {mode} already exists. Not downloaded to {path}")
            continue

        if os.path.exists(extract_path):
            print(f"\nPath for {mode} already exists. Not downloaded to {extract_path}")
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
                print(f"Extracted {mode} to {extract_path}")
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
        "-o",
        "--output-folder",
        dest="output_folder",
        type=str,
        help="Path to the output folder. Default is OneStop",
        default="data/OneStop",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "onestop-full",
            "onestop",
            "repeated",
            "information-seeking",
            "ordinary",
            "information-seeking-in-repeated",
            "asc_files",
            "edf_files",
        ],
        help="Mode of data to download. Options are onestop-full, onestop, ordinary, repeated, "
        "information-seeking, information-seeking-in-repeated. Default is ordinary.",
        default="ordinary",
    )

    args = parser.parse_args()

    download_data(
        extract=args.extract,
        output_folder=args.output_folder,
        mode=args.mode,
    )
