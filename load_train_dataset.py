import argparse
import os
import zipfile
from pathlib import Path

import yadisk

DATASET_URL = "https://disk.360.yandex.ru/d/9k_k6G6a03GURg"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Directory to extract contents")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    y = yadisk.YaDisk()
    meta = y.get_public_meta(DATASET_URL)
    total_size = meta.size
    file_name = meta.name

    zip_path = str(outdir / file_name)
    print(f"Downloading {file_name} ({total_size / 1e6:.2f} MB)")
    y.download_public(DATASET_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(outdir))
    os.remove(zip_path)
    print(f"Files extracted to: {outdir}")


if __name__ == "__main__":
    main()
