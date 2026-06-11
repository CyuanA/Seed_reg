from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image

from src.detector import ImageAnalysis, analyze_image


def load_sample_metadata(path: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(path)
    rows = []
    for _, row in data.sort_values("time_h").iterrows():
        image_path = path.parent / str(row["filename"])
        rows.append(
            {
                "time_h": float(row["time_h"]),
                "path": image_path,
                "expected_total": int(row["total"]),
                "expected_germinated": int(row["germinated"]),
            }
        )
    return rows


def build_summary_table(
    rows: list[dict[str, Any]],
    analyzer: Callable[[Image.Image], ImageAnalysis] = analyze_image,
) -> pd.DataFrame:
    records = []
    for row in rows:
        result = analyzer(Image.open(row["path"]).convert("RGB"))
        records.append(
            {
                "time_h": row["time_h"],
                "total": result.total,
                "germinated": result.germinated,
                "non_germinated": result.non_germinated,
                "germination_rate": round(result.germination_rate, 2),
            }
        )
    return pd.DataFrame(records)


def sequence_rows_from_uploads(uploads, interval_hours: float) -> list[dict[str, Any]]:
    rows = []
    for index, uploaded in enumerate(sorted(uploads, key=lambda item: item.name)):
        rows.append(
            {
                "name": uploaded.name,
                "time_h": round(index * float(interval_hours), 2),
                "image": Image.open(uploaded).convert("RGB"),
            }
        )
    return rows
