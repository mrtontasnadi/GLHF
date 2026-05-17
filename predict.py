"""
Two-stage traffic sign prediction pipeline.

Usage:
    python predict.py --images <folder> --output <folder>
                      --detector  runs/yolo26s_roboflow/.../weights/best.pt
                      --classifier runs/yolo26s_cls_roboflow/.../weights/best.pt
                      [--conf 0.25]

Writes one JSON file per image into the output folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

# ── Label mapping ──────────────────────────────────────────────────────────────

# Mandatory (blue) regulatory signs — everything else is prohibition (red).
_BLUE_REGULATORY_BASES = {
    'regulatory--bicycles-only',
    'regulatory--buses-only',
    'regulatory--dual-path-bicycles-and-pedestrians',
    'regulatory--dual-path-pedestrians-and-bicycles',
    'regulatory--go-straight',
    'regulatory--go-straight-or-turn-left',
    'regulatory--go-straight-or-turn-right',
    'regulatory--keep-left',
    'regulatory--keep-right',
    'regulatory--one-way-left',
    'regulatory--one-way-right',
    'regulatory--one-way-straight',
    'regulatory--pass-on-either-side',
    'regulatory--pedestrians-only',
    'regulatory--priority-over-oncoming-vehicles',
    'regulatory--roundabout',
    'regulatory--shared-path-bicycles-and-pedestrians',
    'regulatory--shared-path-pedestrians-and-bicycles',
    'regulatory--turn-left',
    'regulatory--turn-left-ahead',
    'regulatory--turn-right',
    'regulatory--turn-right-ahead',
}


def reduce_label(fine_label: str) -> str:
    if fine_label.startswith('warning--'):
        return 'Warning'
    if fine_label.startswith('complementary--'):
        return 'complementary'
    if fine_label == 'other-sign':
        return 'other-sign'
    if fine_label.startswith('regulatory--priority-road--'):
        return 'priority road'
    if fine_label.startswith('regulatory--'):
        base = '--'.join(fine_label.split('--')[:2])
        return 'regulatory-blue' if base in _BLUE_REGULATORY_BASES else 'regulatory-red'
    # information--* and anything unrecognised
    return 'other-sign'


# ── Core pipeline ──────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}


def predict_image(
    img_path: Path,
    detector: YOLO,
    classifier: YOLO,
    conf_threshold: float,
) -> list[dict]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'Cannot read image: {img_path}')
    h, w = img.shape[:2]

    det_results = detector(img_path, conf=conf_threshold, verbose=False)[0]

    objects = []
    for box in det_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        # Crop with boundary clamping
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        cls_result = classifier(crop, verbose=False)[0]
        fine_label = cls_result.names[int(cls_result.probs.top1)]

        objects.append({
            'reduced_label': reduce_label(fine_label),
            'confidence': round(confidence, 6),
            'bbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2},
        })

    return objects


def run(
    images_dir: Path,
    output_dir: Path,
    detector_path: Path,
    classifier_path: Path,
    conf: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(detector_path)
    classifier = YOLO(classifier_path)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f'No images found in {images_dir}')
        return

    print(f'Processing {len(image_paths)} images...')
    for img_path in image_paths:
        objects = predict_image(img_path, detector, classifier, conf)
        out_path = output_dir / (img_path.stem + '.json')
        out_path.write_text(json.dumps({'objects': objects}, indent=2), encoding='utf-8')
        print(f'  {img_path.name} → {len(objects)} detection(s)')

    print(f'Done. Results written to {output_dir}')


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Two-stage traffic sign prediction pipeline.')
    parser.add_argument('--images',     required=True, type=Path, help='Input image folder')
    parser.add_argument('--output',     required=True, type=Path, help='Output folder for JSON files')
    parser.add_argument('--detector',   required=True, type=Path, help='Detector weights (.pt)')
    parser.add_argument('--classifier', required=True, type=Path, help='Classifier weights (.pt)')
    parser.add_argument('--conf', type=float, default=0.25, help='Detector confidence threshold (default: 0.25)')
    args = parser.parse_args()

    run(args.images, args.output, args.detector, args.classifier, args.conf)


if __name__ == '__main__':
    main()
