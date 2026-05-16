#!/usr/bin/env python3
"""
Convert a YOLO detection-format dataset (images/ + labels/ per split)
into a classification folder-format dataset (split/class_name/image.jpg).

Assumes each label file has exactly one object: "<class_id> cx cy w h"
(i.e. the whole image is already a cropped sign).

Usage:
    python convert_to_cls_folder.py --src datasets/classification --dst datasets/classification_cls
"""
from __future__ import annotations
import argparse
import shutil
import yaml
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ['train', 'valid', 'test']


def convert(src: Path, dst: Path) -> None:
    data_yaml = src / 'data.yaml'
    if not data_yaml.exists():
        raise FileNotFoundError(f'data.yaml not found in {src}')

    with data_yaml.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    names: list[str] = cfg['names']
    print(f'Classes: {len(names)}, Splits: {SPLITS}')

    for split in SPLITS:
        img_dir = src / split / 'images'
        lbl_dir = src / split / 'labels'
        if not img_dir.exists():
            print(f'  Skipping {split} (no images folder)')
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        print(f'  {split}: {len(images)} images')
        copied = skipped = 0

        for img_path in images:
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                skipped += 1
                continue
            first_line = lbl_path.read_text(encoding='utf-8').strip().splitlines()[0]
            class_id = int(first_line.split()[0])
            class_name = names[class_id]

            out_dir = dst / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, out_dir / img_path.name)
            copied += 1

        print(f'    copied={copied}, skipped={skipped}')

    print(f'\nDone. Output: {dst}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, type=Path)
    ap.add_argument('--dst', required=True, type=Path)
    args = ap.parse_args()
    convert(args.src.resolve(), args.dst.resolve())


if __name__ == '__main__':
    main()
