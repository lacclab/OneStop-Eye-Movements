from pathlib import Path
import subprocess
import argparse
from loguru import logger


def find_edf_files(directory):
    """
    Find all .edf files in the given directory and its subdirectories.

    Parameters:
    - directory (Path): The root directory to search for .edf files.

    Returns:
    - list of Path: List of paths to .edf files.
    """
    edf_files = list(directory.rglob("*.edf"))
    return edf_files


def run_edf2asc(edf_file, output_dir):
    """
    Run the edf2asc command on a given .edf file and save the output in the specified directory.

    Parameters:
    - edf_file (Path): The path to the .edf file.
    - output_dir (Path): The directory to save the output files.
    """
    command = f'edf2asc "{edf_file}" -t -c -v -y -utf8 -p "{output_dir}"'
    try:
        with open("edf2asc.log", "a") as f:
            subprocess.run(
                command, shell=True, check=True, stdout=f, stderr=subprocess.STDOUT
            )
        logger.info(f"Successfully processed {edf_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing {edf_file}: {e}")


def main():
    """
    Main function to parse command line arguments and process .edf files.
    """
    parser = argparse.ArgumentParser(description="Process .edf files using edf2asc.")
    parser.add_argument(
        "root_dir", type=str, help="The root directory to search for .edf files."
    )
    parser.add_argument(
        "output_dir", type=str, help="The directory to save the output files."
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.add("edf2asc_wrapper.log", rotation="1 MB")

    edf_files = find_edf_files(root_dir)

    if not edf_files:
        logger.warning("No .edf files found in the specified directory.")
    else:
        logger.info(f"Found {len(edf_files)} .edf files.")
        for edf_file in edf_files:
            logger.info(f"Processing {edf_file}...")
            run_edf2asc(edf_file, output_dir)
            logger.info("Done.")


if __name__ == "__main__":
    main()
