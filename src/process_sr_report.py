"""This script is used to parse the raw data and save it in a format
that is easier to work with."""

from pathlib import Path

import process_onestop_sr_report.preprocessing as prp
from datetime import datetime


def process_data(mode: str):
    data_path = "/Users/shubi/Data/l2_report/l2_report/Output/ia_p.tsv"

    today = datetime.today().strftime("%d%m%Y")
    save_file = f"{mode}_l2_p_{today}.csv"
    args_file = f"{mode}_l2_p_{today}_args.json"
    save_path = Path("data/Outputs")
    args = [
        "--data_path",
        data_path,
        "--save_path",
        str(save_path / save_file),
        "--mode",
        mode,
        "--filter_query",
        "practice==0",
    ]

    cfg = prp.ArgsParser().parse_args(args)

    args_save_path = save_path / args_file
    save_path.mkdir(parents=True, exist_ok=True)
    cfg.save(str(args_save_path))
    print(f"Saved config to {args_save_path}")

    print(f"Running preprocessing with args: {args}")
    prp.preprocess_data(cfg)


if __name__ == "__main__":
    process_data(prp.Mode.IA.value)
    # process_data(prp.Mode.FIXATION.value)