#!/usr/bin/env python3
"""
Convert per-image JSON annotations to YOLOv10+ compatible dataset layout
while preserving every original annotation field/value in sidecar metadata.

Input expected:
- images_dir: folder with .jpg/.png images (e.g. train/)
- ann_dir: folder with JSON files named like <image_stem>.json
          (e.g. train/train/annotations/)

Output layout:
out_dir/
  images/
  labels/
  meta/
  data.yaml
  classes.json
  annotations_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def yolo_from_xyxy(
    xmin: float, ymin: float, xmax: float, ymax: float, img_w: float, img_h: float
) -> Tuple[float, float, float, float]:
    # Standard YOLO normalized center-width-height conversion.
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def convert(images_dir: Path, ann_dir: Path, out_dir: Path) -> None:
    out_images = out_dir / "images"
    out_labels = out_dir / "labels"
    out_meta = out_dir / "meta"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    # Collect image files by stem for matching.
    stem_to_image: Dict[str, Path] = {}
    for p in images_dir.iterdir():
        print("Checking file:", p)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            stem_to_image[p.stem] = p
            print("Found image:", p)

    # Read all annotation json files.
    ann_files = sorted([p for p in ann_dir.glob("*.json") if p.is_file()])

    label_to_id: Dict[str, int] = {}
    full_records: List[dict] = []

    for ann_path in ann_files:
        stem = ann_path.stem
        if stem not in stem_to_image:
            # Skip annotations whose image is missing.
            print(f"Warning: No image found for annotation {ann_path.name}, skipping.")
            continue

        img_path = stem_to_image[stem]
        raw = json.loads(ann_path.read_text(encoding="utf-8"))

        # Preserve full original annotation exactly as loaded.
        full_records.append(
            {
                "image_stem": stem,
                "image_file": img_path.name,
                "annotation_file": ann_path.name,
                "original": raw,
            }
        )

        # Copy image unchanged.
        shutil.copy2(img_path, out_images / img_path.name)

        # Save exact original JSON as sidecar metadata.
        (out_meta / ann_path.name).write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        w = raw["width"]
        h = raw["height"]
        objects = raw.get("objects", [])

        yolo_lines: List[str] = []

        for obj in objects:
            label = obj["label"]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)

            cid = label_to_id[label]
            bbox = obj["bbox"]
            x, y, bw, bh = yolo_from_xyxy(
                float(bbox["xmin"]),
                float(bbox["ymin"]),
                float(bbox["xmax"]),
                float(bbox["ymax"]),
                float(w),
                float(h),
            )

            # Keep numeric precision high but concise.
            yolo_lines.append(f"{cid} {x:.10f} {y:.10f} {bw:.10f} {bh:.10f}")

        (out_labels / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    # Save class mapping (id -> label) for deterministic import/traceability.
    classes = [{"id": i, "name": name} for name, i in sorted(label_to_id.items(), key=lambda kv: kv[1])]
    (out_dir / "classes.json").write_text(
        json.dumps(classes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # YOLO data config (single split package; Roboflow/CVAT can still import).
    names = [c["name"] for c in classes]
    data_yaml = "\n".join(
        [
            f"path: {out_dir.resolve().as_posix()}",
            "train: images",
            "val: images",
            f"nc: {len(names)}",
            "names:",
            *[f"  {i}: {n}" for i, n in enumerate(names)],
            "",
        ]
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # Preserve all original annotation payloads in one stream file.
    with (out_dir / "annotations_full.jsonl").open("w", encoding="utf-8") as f:
        for rec in full_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. Converted {len(full_records)} images with annotations.")
    print(f"Output: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, type=Path, help="Folder with source images")
    ap.add_argument("--ann-dir", required=True, type=Path, help="Folder with source JSON annotations")
    ap.add_argument("--out-dir", required=True, type=Path, help="Output dataset folder")
    args = ap.parse_args()

    convert(args.images_dir, args.ann_dir, args.out_dir)


if __name__ == "__main__":
    main()