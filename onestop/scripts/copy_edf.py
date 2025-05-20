import shutil
from pathlib import Path


def copy_edf_files(src_dir: str, dst_dir: str, preserve_structure: bool = True):
    """
    Copy all .edf files from src_dir to dst_dir.

    :param src_dir: Path to the root folder to search.
    :param dst_dir: Path to the destination folder.
    :param preserve_structure:
        - If True, recreates the subfolder layout under dst_dir.
        - If False, copies all .edf files directly into dst_dir (flat).
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    for edf_path in src.rglob("*.edf"):
        if preserve_structure:
            # build the same relative path under dst
            rel_path = edf_path.relative_to(src)
            target_path = dst / rel_path
        else:
            # place all files directly under dst
            target_path = dst / edf_path.name

        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(edf_path, target_path)
        print(f"Copied {edf_path} → {target_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively copy all .edf files from one folder to another."
    )
    parser.add_argument("src", help="Source directory to search for .edf files")
    parser.add_argument("dst", help="Destination directory for copied files")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="If set, copies all .edf files into dst without subdirectories",
    )
    args = parser.parse_args()

    copy_edf_files(args.src, args.dst, preserve_structure=not args.flat)
