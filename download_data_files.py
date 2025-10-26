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
        mode (str): The mode of data to download. Options are 'onestop-full', 'onestop_all_regimes', 'repeated',
                    'information_seeking', 'ordinary', 'information_seeking_repeated'. Default is 'ordinary'.

    Returns:
        None
    """
    base_url = "https://osf.io/download/"

    urls = {
        "ordinary": {"fixations_Paragraph": "ne4az", "ia_Paragraph": "xkgfz"},
        "information_seeking": {
            "fixations_Paragraph": "bznfk",
            "ia_Paragraph": "yxzte",
        },
        "repeated": {"fixations_Paragraph": "83ctd", "ia_Paragraph": "dwfk4"},
        "information_seeking_repeated": {
            "fixations_Paragraph": "paqn8",
            "ia_Paragraph": "ygjup",
        },
        "onestop_all_regimes": {"fixations_Paragraph": "dq935", "ia_Paragraph": "4ajc8"},
        "onestop-full": {
            "fixations_title": "uwz2e",
            "ia_title": "u7f9b",
            "fixations_question_preview": "7a3md",
            "ia_question_preview": "zn473",
            "fixations_paragraph": "tbxdc",
            "ia_paragraph": "zhywq",
            "fixations_questions": "cmx6k",
            "ia_questions": "tcv9h",
            "fixations_answers": "ax4md",
            "ia_answers": "q3shp",
            "fixations_qa": "fg7se",
            "ia_qa": "3j8av",
            "fixations_feedback": "e76vz",
            "ia_feedback": "t6n8v",
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
        req = requests.get(url, stream=True)

        # create new paths for the downloaded files
        subcorpora = [
            "repeated",
            "information_seeking",
            "ordinary",
            "information_seeking_repeated",
        ]
        if mode in subcorpora:
            resource_name = f"{resource_name}_{mode}"
        filename = f"{resource_name}.csv.zip"
        path = folder / mode / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"\nPath for {path} already exists. Not downloaded.")
            continue
        print(f"Downloading {mode} - {resource_name} ({url}) to {path}")

        # Writing the file to the local file system
        with open(path, "wb") as output_file:
            for chunk in req.iter_content(chunk_size=128):
                output_file.write(chunk)

        if extract:
            extract_path = folder / mode
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
        default=False,
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
            "information_seeking",
            "ordinary",
            "information_seeking_repeated",
        ],
        help="Mode of data to download. Options are onestop-full, onestop_all_regimes, ordinary, repeated, "
        "information_seeking, information_seeking_repeated. Default is ordinary.",
        default="ordinary",
    )

    args = parser.parse_args()

    download_data(
        extract=args.extract,
        output_folder=args.output_folder,
        mode=args.mode,
    )
