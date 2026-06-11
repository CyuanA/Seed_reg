from __future__ import annotations

import argparse
import csv
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CLASS_NAMES = ["non_germinated", "germinated"]


def class_id(raw_name: str) -> int:
    if raw_name.endswith("_im"):
        return 0
    if raw_name.endswith("_el"):
        return 1
    raise ValueError(f"Unknown class name: {raw_name}")


def petri_prefix(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    return "_".join(parts[:2])


def parse_voc(xml_bytes: bytes) -> tuple[str, int, int, list[tuple[int, float, float, float, float]]]:
    root = ET.fromstring(xml_bytes)
    filename = root.findtext("filename")
    size = root.find("size")
    if filename is None or size is None:
        raise ValueError("Invalid VOC XML: missing filename or size")

    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if name is None or bnd is None:
            continue
        xmin = max(0.0, float(bnd.findtext("xmin")))
        ymin = max(0.0, float(bnd.findtext("ymin")))
        xmax = min(float(width), float(bnd.findtext("xmax")))
        ymax = min(float(height), float(bnd.findtext("ymax")))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        boxes.append((class_id(name), x_center, y_center, box_width, box_height))
    return filename, width, height, boxes


def split_prefixes(prefixes: list[str], seed: int, train_ratio: float, val_ratio: float) -> dict[str, str]:
    shuffled = list(prefixes)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = max(1, int(n_total * train_ratio))
    n_val = max(1, int(n_total * val_ratio))
    mapping = {}
    for index, prefix in enumerate(shuffled):
        if index < n_train:
            mapping[prefix] = "train"
        elif index < n_train + n_val:
            mapping[prefix] = "val"
        else:
            mapping[prefix] = "test"
    return mapping


def clean_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ["train", "val", "test"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/public_dataset/raw/GermPredDataset.zip")
    parser.add_argument("--species", default="ZeaMays", choices=["ZeaMays", "SecaleCereale", "PennisetumGlaucum"])
    parser.add_argument("--output", default="data/public_dataset/yolo_seed")
    parser.add_argument("--max-prefixes", type=int, default=36, help="Limit petri-dish sequences for a fast first training run.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    output = Path(args.output)
    clean_output(output)

    with zipfile.ZipFile(zip_path) as archive:
        xml_names = [
            name for name in archive.namelist()
            if name.startswith(f"GermPredDataset/{args.species}/true_ann/") and name.lower().endswith(".xml")
        ]
        img_names = {
            Path(name).name: name
            for name in archive.namelist()
            if name.startswith(f"GermPredDataset/{args.species}/img/") and name.lower().endswith(".jpg")
        }

        by_prefix: dict[str, list[str]] = defaultdict(list)
        for xml_name in xml_names:
            by_prefix[petri_prefix(Path(xml_name).stem)].append(xml_name)

        prefixes = sorted(by_prefix)
        if args.max_prefixes and args.max_prefixes < len(prefixes):
            prefixes = sorted(random.Random(args.seed).sample(prefixes, args.max_prefixes))
        split_map = split_prefixes(prefixes, args.seed, args.train_ratio, args.val_ratio)

        metadata_rows = []
        split_counter = Counter()
        object_counter = Counter()

        for prefix in prefixes:
            split = split_map[prefix]
            for xml_name in sorted(by_prefix[prefix]):
                filename, width, height, boxes = parse_voc(archive.read(xml_name))
                image_name = img_names.get(filename)
                if image_name is None:
                    continue

                stem = Path(filename).stem
                image_out = output / "images" / split / filename
                label_out = output / "labels" / split / f"{stem}.txt"

                image_out.write_bytes(archive.read(image_name))
                with label_out.open("w", encoding="utf-8") as file:
                    for box in boxes:
                        file.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*box))

                counts = Counter(box[0] for box in boxes)
                metadata_rows.append(
                    {
                        "split": split,
                        "species": args.species,
                        "prefix": prefix,
                        "filename": filename,
                        "stem": stem,
                        "width": width,
                        "height": height,
                        "non_germinated": counts[0],
                        "germinated": counts[1],
                        "total": len(boxes),
                    }
                )
                split_counter[split] += 1
                object_counter.update(counts)

    with (output / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split", "species", "prefix", "filename", "stem",
                "width", "height", "non_germinated", "germinated", "total",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    yaml_text = "\n".join(
        [
            "path: .",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: non_germinated",
            "  1: germinated",
            "",
        ]
    )
    (output / "seed_germination.yaml").write_text(yaml_text, encoding="utf-8")

    print("Prepared YOLO dataset:", output.resolve())
    print("Species:", args.species)
    print("Petri-dish prefixes:", len(prefixes))
    print("Images by split:", dict(split_counter))
    print("Objects:", {CLASS_NAMES[key]: value for key, value in object_counter.items()})


if __name__ == "__main__":
    main()
