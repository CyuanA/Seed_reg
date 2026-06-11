from __future__ import annotations

import argparse
from pathlib import Path

from curl_cffi import requests


DATASET_ID = "4wkt6thgp6"
VERSION = 3
REFERER = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
FILES_URL = (
    f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
    f"?folder_id=root&version={VERSION}"
)


def get_files() -> list[dict]:
    response = requests.get(
        FILES_URL,
        impersonate="chrome120",
        timeout=60,
        headers={
            "Accept": "application/vnd.mendeley-public-dataset.1+json",
            "Referer": REFERER,
        },
    )
    response.raise_for_status()
    return response.json()


def download_file(url: str, output_path: Path, expected_size: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.stat().st_size if output_path.exists() else 0

    headers = {"Referer": REFERER}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    response = requests.get(
        url,
        impersonate="chrome120",
        stream=True,
        timeout=120,
        headers=headers,
    )
    response.raise_for_status()
    mode = "ab" if existing and response.status_code == 206 else "wb"
    if mode == "wb":
        existing = 0

    downloaded = existing
    with output_path.open(mode) as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            downloaded += len(chunk)
            if expected_size:
                percent = downloaded / expected_size * 100
                print(
                    f"\r{output_path.name}: {downloaded / 1e6:.1f} MB / "
                    f"{expected_size / 1e6:.1f} MB ({percent:.1f}%)",
                    end="",
                )
            else:
                print(f"\r{output_path.name}: {downloaded / 1e6:.1f} MB", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="GermPredDataset.zip")
    parser.add_argument("--out-dir", default="data/public_dataset/raw")
    args = parser.parse_args()

    files = get_files()
    selected = next((item for item in files if item["filename"] == args.filename), None)
    if selected is None:
        available = ", ".join(item["filename"] for item in files)
        raise SystemExit(f"Cannot find {args.filename}. Available files: {available}")

    output_path = Path(args.out_dir) / selected["filename"]
    download_file(
        selected["content_details"]["download_url"],
        output_path,
        int(selected.get("size") or selected["content_details"].get("size") or 0),
    )


if __name__ == "__main__":
    main()
