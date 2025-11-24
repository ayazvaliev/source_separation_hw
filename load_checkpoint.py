import argparse
import zipfile
from pathlib import Path

import gdown

CKPT_URL = "https://drive.google.com/file/d/1-ts2Oe_f4zAEuiEyIsAq2Gn9VTlt8S9T/view?usp=sharing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Directory to extract contents")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    zip_path = outdir / "ckpt.zip"

    print("Downloading file from GDrive...")
    gdown.download(CKPT_URL, str(zip_path), quiet=False, fuzzy=True)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(outdir))

    zip_path.unlink()
    print(f"Files extracted to: {outdir}")


if __name__ == "__main__":
    main()
