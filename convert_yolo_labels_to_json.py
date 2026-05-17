"""
Convert YOLO detection labels (class_id cx cy w h, normalised) to the GT JSON
format expected by evaluate.py.

Reads:  <labels_dir>/<stem>.txt  +  <images_dir>/<stem>.<ext>
Writes: <output_dir>/<stem>.json

Usage:
    python convert_yolo_labels_to_json.py
        --images  datasets/test/images
        --labels  datasets/test/labels
        --yaml    datasets/data.yaml
        --output  datasets/test/annotations
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def convert(images_dir: Path, labels_dir: Path, class_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f'No images found in {images_dir}')
        return

    print(f'Converting {len(image_paths)} label files...')
    missing = 0

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            print(f'  WARNING: no label file for {img_path.name}, skipping')
            missing += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'  WARNING: cannot read {img_path.name}, skipping')
            missing += 1
            continue
        h, w = img.shape[:2]

        objects = []
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            xmin = (cx - bw / 2) * w
            ymin = (cy - bh / 2) * h
            xmax = (cx + bw / 2) * w
            ymax = (cy + bh / 2) * h
            objects.append({
                'label': class_names[cls_id],
                'bbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
            })

        out_path = output_dir / (img_path.stem + '.json')
        out_path.write_text(json.dumps({'objects': objects}, indent=2), encoding='utf-8')
        print(f'  {img_path.name} → {len(objects)} object(s)')

    print(f'Done. {len(image_paths) - missing} files written to {output_dir}')
    if missing:
        print(f'  {missing} file(s) skipped (missing label or unreadable image).')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert YOLO labels to GT annotation JSONs.')
    parser.add_argument('--images', required=True, type=Path, help='Folder of images')
    parser.add_argument('--labels', required=True, type=Path, help='Folder of YOLO .txt label files')
    parser.add_argument('--yaml',   required=True, type=Path, help='data.yaml with class names')
    parser.add_argument('--output', required=True, type=Path, help='Output folder for JSON files')
    args = parser.parse_args()

    data = yaml.safe_load(args.yaml.read_text(encoding='utf-8'))
    class_names: list[str] = data['names']

    convert(args.images, args.labels, class_names, args.output)


if __name__ == '__main__':
    main()
