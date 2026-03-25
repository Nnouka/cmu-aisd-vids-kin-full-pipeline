#!/usr/bin/env python3
"""Download the DeepKIN TTS model using the signed URL API flow.

This script mirrors the notebook logic used in tts-models-deepkin.ipynb:
1) Request a signed GET URL for kinya_flex_tts_base_trained.pt
2) Download and write the file locally
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import requests


# Make backend package importable when running this script from repo root.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.aws_gateway import generate_signed_url


DEFAULT_BUCKET = "cmu-aisd-output"
DEFAULT_KEY = "kinya_flex_tts_base_trained.pt"
DEFAULT_CONTENT_TYPE = "application/octet-stream"
DEFAULT_OUTPUT = "kinya_flex_tts_base_trained.pt"


def download_file(url: str, output_path: pathlib.Path, chunk_size: int = 8192) -> None:
    with requests.get(url, stream=True, timeout=600) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download model file: {response.status_code} - {response.text}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download kinya_flex_tts_base_trained.pt from S3 via signed URL")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket containing the model")
    parser.add_argument("--key", default=DEFAULT_KEY, help="S3 key for the model file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Local path to save the model")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = pathlib.Path(args.output).resolve()

    if output_path.exists() and not args.force:
        print(f"Model file already exists at: {output_path}")
        print("Use --force to overwrite.")
        return 0

    try:
        print("Requesting signed URL for model download...")
        signed = generate_signed_url(
            s3_key=args.key,
            bucket=args.bucket,
            action="get_object",
            content_type=DEFAULT_CONTENT_TYPE,
        )

        print(f"Downloading model to: {output_path}")
        download_file(signed["url"], output_path)
        print("Download complete.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
