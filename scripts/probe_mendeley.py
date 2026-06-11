from __future__ import annotations

import re
from pathlib import Path

from curl_cffi import requests


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "mendeley_page.html"
BUNDLE_PATH = ROOT / "mendeley_bundle.js"


def fetch(url: str) -> str:
    response = requests.get(
        url,
        impersonate="chrome120",
        timeout=60,
        headers={"referer": "https://data.mendeley.com/datasets/4wkt6thgp6/3"},
    )
    response.raise_for_status()
    return response.text


def main() -> None:
    html = fetch("https://data.mendeley.com/datasets/4wkt6thgp6/3")
    HTML_PATH.write_text(html, encoding="utf-8")

    scripts = re.findall(r'<script src="([^"]*bundle\.js[^"]*)"', html)
    print("bundle scripts:", scripts)

    for script in scripts:
        url = "https://data.mendeley.com" + script if script.startswith("/") else script
        bundle = fetch(url)
        BUNDLE_PATH.write_text(bundle, encoding="utf-8")
        print("downloaded bundle", url, len(bundle))

    bundle = BUNDLE_PATH.read_text(encoding="utf-8")
    patterns = [
        "api/datasets",
        "files",
        "download",
        "folders",
        "archive",
        "public-api",
    ]
    for pattern in patterns:
        print(f"\nPATTERN {pattern}")
        for match in re.finditer(pattern, bundle, re.IGNORECASE):
            start = max(0, match.start() - 180)
            end = min(len(bundle), match.end() + 260)
            print(bundle[start:end])
            print("---")


if __name__ == "__main__":
    main()
