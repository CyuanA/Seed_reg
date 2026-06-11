from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "sample_sequence"


def draw_seed(canvas, center, angle_deg, germinated=False, sprout_length=0):
    cx, cy = center
    axes = (18, 10)
    color = (125, 164, 198)
    cv2.ellipse(canvas, (cx, cy), axes, angle_deg, 0, 360, color, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, cy), axes, angle_deg, 0, 360, (92, 120, 145), 1, cv2.LINE_AA)

    if germinated and sprout_length > 0:
        radians = np.deg2rad(angle_deg - 35)
        start = (
            int(cx + np.cos(radians) * 13),
            int(cy + np.sin(radians) * 9),
        )
        end = (
            int(start[0] + np.cos(radians) * sprout_length),
            int(start[1] + np.sin(radians) * sprout_length),
        )
        cv2.line(canvas, start, end, (72, 132, 92), 5, cv2.LINE_AA)
        cv2.circle(canvas, end, 3, (64, 120, 84), -1, cv2.LINE_AA)


def make_frame(time_index: int, time_h: int, germinated_count: int):
    canvas = np.full((620, 620, 3), (244, 244, 238), dtype=np.uint8)
    cv2.circle(canvas, (310, 310), 280, (220, 220, 214), 4, cv2.LINE_AA)
    cv2.circle(canvas, (310, 310), 268, (248, 248, 244), -1, cv2.LINE_AA)

    positions = [
        (205, 150), (315, 135), (430, 160),
        (155, 275), (280, 260), (410, 275), (500, 300),
        (190, 410), (320, 430), (450, 405),
    ]
    angles = [-18, 22, -45, 35, -8, 18, -28, 42, -35, 12]

    rng = np.random.default_rng(100 + time_index)
    for index, (center, angle) in enumerate(zip(positions, angles)):
        jitter = rng.integers(-4, 5, size=2)
        shifted = (int(center[0] + jitter[0]), int(center[1] + jitter[1]))
        germinated = index < germinated_count
        sprout_length = max(0, 18 + time_index * 7 + index % 3 * 4) if germinated else 0
        draw_seed(canvas, shifted, angle, germinated, sprout_length)

    cv2.putText(canvas, f"{time_h} h", (32, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (85, 85, 85), 2, cv2.LINE_AA)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule = [
        (0, 0),
        (6, 1),
        (12, 3),
        (18, 5),
        (24, 7),
        (30, 8),
        (36, 9),
        (42, 9),
    ]

    metadata = []
    for index, (time_h, germinated_count) in enumerate(schedule):
        image = make_frame(index, time_h, germinated_count)
        filename = f"sample_{index:03d}_{time_h:02d}h.png"
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError(f"Failed to encode {filename}")
        encoded.tofile(OUT_DIR / filename)
        metadata.append(
            {
                "filename": filename,
                "time_h": time_h,
                "total": 10,
                "germinated": germinated_count,
            }
        )

    with (OUT_DIR / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "time_h", "total", "germinated"])
        writer.writeheader()
        writer.writerows(metadata)


if __name__ == "__main__":
    main()
