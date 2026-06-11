from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASS_MAP = {
    "non_germinated": 0,
    "germinated": 1,
    "seeds_im": 0,
    "seed_im": 0,
    "seeds_el": 1,
    "seed_el": 1,
}


def normalize_class_name(name: str) -> int:
    raw = name.strip()
    if raw in CLASS_MAP:
        return CLASS_MAP[raw]
    if raw.endswith("_im"):
        return 0
    if raw.endswith("_el"):
        return 1
    raise ValueError(f"Unknown VOC class name: {name}")


def convert_xml(xml_path: Path) -> tuple[str, list[str]]:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    if filename is None or size is None:
        raise ValueError(f"Invalid VOC annotation: {xml_path}")

    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in annotation: {xml_path}")

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        box = obj.find("bndbox")
        if name is None or box is None:
            continue

        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(width, float(box.findtext("xmax", "0")))
        ymax = min(height, float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        labels.append(
            f"{normalize_class_name(name)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    return filename, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Pascal VOC seed annotations to YOLO format.")
    parser.add_argument("--annotations", required=True, help="Directory containing VOC XML annotations.")
    parser.add_argument("--images", required=True, help="Directory containing source images.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--copy-images", action="store_true", help="Copy matching images into output/images.")
    args = parser.parse_args()

    annotations = Path(args.annotations)
    images = Path(args.images)
    output = Path(args.output)
    labels_out = output / "labels"
    images_out = output / "images"
    labels_out.mkdir(parents=True, exist_ok=True)
    if args.copy_images:
        images_out.mkdir(parents=True, exist_ok=True)

    converted = 0
    for xml_path in sorted(annotations.glob("*.xml")):
        filename, labels = convert_xml(xml_path)
        (labels_out / f"{Path(filename).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        if args.copy_images:
            src = images / filename
            if src.exists():
                shutil.copy2(src, images_out / src.name)
        converted += 1

    print(f"Converted {converted} annotation files to {labels_out}")


if __name__ == "__main__":
    main()
