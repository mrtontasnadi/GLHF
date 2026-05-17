"""
Quality and runtime evaluation for the two-stage prediction pipeline.

Runs inference on a test image folder with timing instrumentation, compares
the predictions against ground-truth annotation JSONs, and reports:
  - Per-class: precision, recall, F1, AP
  - Overall:   mAP@IoU=0.5
  - Runtime:   detector ms, classifier ms, total ms, throughput, GPU memory

Usage:
    python evaluate.py
        --images      <test_image_folder>
        --gt          <ground_truth_annotation_folder>
        --output      <folder_for_prediction_jsons_and_eval_results>
        --detector    <detector_weights.pt>
        --classifier  <classifier_weights.pt>
        [--conf 0.25]
        [--iou-threshold 0.5]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict import IMAGE_EXTENSIONS, reduce_label

# ── Stem mapping ──────────────────────────────────────────────────────────────

def gt_stem(img_stem: str) -> str:
    """Map a Roboflow image stem to its original annotation stem.

    Roboflow appends '_jpg.rf.<hash>' to original filenames, e.g.:
      0p7096-UnLzQBO-1Cu5Atw_jpg.rf.e467d5...  ->  0p7096-UnLzQBO-1Cu5Atw
    If no such suffix is present the stem is returned unchanged.
    """
    if '_jpg' in img_stem:
        return img_stem.split('_jpg')[0]
    return img_stem

# ── Ground-truth loading ───────────────────────────────────────────────────────

def load_ground_truth(gt_dir: Path) -> dict[str, list[dict]]:
    """Load GT annotation JSONs and convert fine-grained labels to reduced_label."""
    gt: dict[str, list[dict]] = {}
    for json_path in sorted(gt_dir.glob('*.json')):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        key = gt_stem(json_path.stem)  # normalise so both GT and preds use the same key
        gt[key] = [
            {'reduced_label': reduce_label(obj['label']), 'bbox': obj['bbox']}
            for obj in data.get('objects', [])
        ]
    return gt

# ── IoU ───────────────────────────────────────────────────────────────────────

def iou(b1: dict, b2: dict) -> float:
    ix1 = max(b1['xmin'], b2['xmin'])
    iy1 = max(b1['ymin'], b2['ymin'])
    ix2 = min(b1['xmax'], b2['xmax'])
    iy2 = min(b1['ymax'], b2['ymax'])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1['xmax'] - b1['xmin']) * (b1['ymax'] - b1['ymin'])
    a2 = (b2['xmax'] - b2['xmin']) * (b2['ymax'] - b2['ymin'])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

# ── AP calculation ────────────────────────────────────────────────────────────

def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    """Area under the precision-recall curve via 101-point interpolation (COCO style)."""
    r = np.array([0.0] + recalls + [1.0])
    p = np.array([1.0] + precisions + [0.0])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        pts = p[r >= t]
        ap += pts.max() if len(pts) else 0.0
    return ap / 101

# ── Quality evaluation ────────────────────────────────────────────────────────

REDUCED_LABELS = [
    'Warning', 'regulatory-blue', 'regulatory-red',
    'complementary', 'priority road', 'other-sign',
]


def evaluate_quality(
    all_preds: dict[str, list[dict]],
    all_gts:   dict[str, list[dict]],
    iou_thresh: float,
) -> dict:
    """
    Compute per-class precision, recall, F1, AP and overall mAP.

    Predictions must have keys 'reduced_label', 'confidence', 'bbox'.
    Ground truths must have keys 'reduced_label', 'bbox'.
    """
    # Per class: list of (confidence, is_tp)
    class_dets:  dict[str, list[tuple[float, bool]]] = defaultdict(list)
    class_n_gt:  dict[str, int] = defaultdict(int)

    for stem in set(all_preds) | set(all_gts):
        preds = sorted(all_preds.get(stem, []), key=lambda x: x['confidence'], reverse=True)
        gts   = all_gts.get(stem, [])
        matched_gt: set[int] = set()

        for pred in preds:
            cls = pred['reduced_label']
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if gt['reduced_label'] != cls or j in matched_gt:
                    continue
                v = iou(pred['bbox'], gt['bbox'])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_thresh:
                matched_gt.add(best_j)
                class_dets[cls].append((pred['confidence'], True))
            else:
                class_dets[cls].append((pred['confidence'], False))

        for gt in gts:
            class_n_gt[gt['reduced_label']] += 1

    results: dict = {}
    aps: list[float] = []

    for cls in REDUCED_LABELS:
        n_gt = class_n_gt[cls]
        dets = sorted(class_dets[cls], key=lambda x: x[0], reverse=True)

        if dets:
            tp_cum = np.cumsum([1 if tp else 0 for _, tp in dets])
            fp_cum = np.cumsum([0 if tp else 1 for _, tp in dets])
            tp = int(tp_cum[-1])
            fp = int(fp_cum[-1])
            recalls    = (tp_cum / n_gt).tolist() if n_gt else [0.0] * len(dets)
            precisions = (tp_cum / (tp_cum + fp_cum)).tolist()
        else:
            tp = fp = 0
            recalls = precisions = []

        fn        = max(0, n_gt - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ap        = compute_ap(recalls, precisions) if n_gt > 0 else 0.0

        results[cls] = {
            'n_gt': n_gt, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': round(precision, 4),
            'recall':    round(recall,    4),
            'f1':        round(f1,        4),
            'ap':        round(ap,        4),
        }
        if n_gt > 0:
            aps.append(ap)

    results['mAP'] = round(float(np.mean(aps)) if aps else 0.0, 4)
    return results

# ── Timed inference ───────────────────────────────────────────────────────────

def predict_image_timed(
    img_path:   Path,
    detector:   YOLO,
    classifier: YOLO,
    conf:       float,
) -> tuple[list[dict], dict]:
    """Run the full pipeline on one image and return (predictions, timing_ms)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'Cannot read image: {img_path}')
    h, w = img.shape[:2]

    t0 = time.perf_counter()
    det_results = detector(img_path, conf=conf, verbose=False)[0]
    t_det = (time.perf_counter() - t0) * 1000

    objects: list[dict] = []
    t_cls = 0.0

    for box in det_results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])

        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        t1 = time.perf_counter()
        cls_result = classifier(crop, verbose=False)[0]
        t_cls += (time.perf_counter() - t1) * 1000

        fine_label = cls_result.names[int(cls_result.probs.top1)]
        objects.append({
            'reduced_label': reduce_label(fine_label),
            'confidence':    round(confidence, 6),
            'bbox': {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2},
        })

    timing = {
        'detector_ms':   round(t_det, 2),
        'classifier_ms': round(t_cls, 2),
        'total_ms':      round(t_det + t_cls, 2),
        'n_detections':  len(objects),
    }
    return objects, timing

# ── Console report ────────────────────────────────────────────────────────────

def print_report(quality: dict, runtime: dict) -> None:
    W = 74
    print('\n' + '=' * W)
    print(' QUALITY METRICS  (IoU threshold used for TP matching shown in header)')
    print('=' * W)
    hdr = f"{'Class':<22} {'GT':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AP':>7}"
    print(hdr)
    print('-' * W)
    for cls in REDUCED_LABELS:
        m = quality[cls]
        print(
            f"{cls:<22} {m['n_gt']:>5} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
            f" {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} {m['ap']:>7.3f}"
        )
    print('-' * W)
    print(f"{'mAP':<22} {'':>35} {quality['mAP']:>7.3f}")

    print('\n' + '=' * W)
    print(' RUNTIME METRICS')
    print('=' * W)
    rows = [
        ('Images processed',          runtime['n_images']),
        ('Total time (s)',             runtime['total_time_s']),
        ('Throughput (img/s)',         runtime['throughput_img_per_s']),
        ('Mean total time / img (ms)', runtime['mean_total_ms']),
        ('Std total time / img (ms)',  runtime['std_total_ms']),
        ('Min total time / img (ms)',  runtime['min_total_ms']),
        ('Max total time / img (ms)',  runtime['max_total_ms']),
        ('Mean detector time (ms)',    runtime['mean_detector_ms']),
        ('Mean classifier time (ms)',  runtime['mean_classifier_ms']),
        ('Mean detections / img',      runtime['mean_detections_per_image']),
    ]
    if 'gpu_peak_memory_mb' in runtime:
        rows.append(('GPU peak memory (MB)', runtime['gpu_peak_memory_mb']))
    for label, value in rows:
        print(f'  {label:<35} {value}')
    print('=' * W + '\n')

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate the two-stage traffic sign pipeline.')
    parser.add_argument('--images',        required=True, type=Path, help='Test image folder')
    parser.add_argument('--gt',            required=True, type=Path, help='Ground-truth annotation folder')
    parser.add_argument('--output',        required=True, type=Path, help='Output folder (predictions + eval JSON)')
    parser.add_argument('--detector',      required=True, type=Path, help='Detector weights (.pt)')
    parser.add_argument('--classifier',    required=True, type=Path, help='Classifier weights (.pt)')
    parser.add_argument('--conf',          type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou-threshold', type=float, default=0.5,  dest='iou_thresh',
                        help='IoU threshold for TP matching (default: 0.5)')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    detector   = YOLO(args.detector)
    classifier = YOLO(args.classifier)

    print('Loading ground truth...')
    all_gts = load_ground_truth(args.gt)
    print(f'  {len(all_gts)} annotation files found.')

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f'No images found in {args.images}')
        return

    print(f'Running timed inference on {len(image_paths)} images...')
    all_preds: dict[str, list[dict]] = {}
    timings:   list[dict]            = []

    for img_path in image_paths:
        objects, timing = predict_image_timed(img_path, detector, classifier, args.conf)
        all_preds[gt_stem(img_path.stem)] = objects
        timings.append(timing)
        out_path = args.output / (img_path.stem + '.json')
        out_path.write_text(json.dumps({'objects': objects}, indent=2), encoding='utf-8')
        print(f'  {img_path.name:<40} {timing["n_detections"]:>3} det  '
              f'{timing["detector_ms"]:>7.1f}ms det  {timing["classifier_ms"]:>7.1f}ms cls')

    # Runtime summary
    det_ms = [t['detector_ms']   for t in timings]
    cls_ms = [t['classifier_ms'] for t in timings]
    tot_ms = [t['total_ms']      for t in timings]
    n_det  = [t['n_detections']  for t in timings]

    runtime = {
        'n_images':                  len(image_paths),
        'total_time_s':              round(sum(tot_ms) / 1000, 2),
        'throughput_img_per_s':      round(1000 * len(image_paths) / sum(tot_ms), 2) if sum(tot_ms) else 0.0,
        'mean_total_ms':             round(float(np.mean(tot_ms)), 2),
        'std_total_ms':              round(float(np.std(tot_ms)),  2),
        'min_total_ms':              round(float(np.min(tot_ms)),  2),
        'max_total_ms':              round(float(np.max(tot_ms)),  2),
        'mean_detector_ms':          round(float(np.mean(det_ms)), 2),
        'mean_classifier_ms':        round(float(np.mean(cls_ms)), 2),
        'mean_detections_per_image': round(float(np.mean(n_det)),  2),
    }

    try:
        import torch
        if torch.cuda.is_available():
            runtime['gpu_peak_memory_mb'] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    except Exception:
        pass

    print('Computing quality metrics...')
    quality = evaluate_quality(all_preds, all_gts, args.iou_thresh)

    print_report(quality, runtime)

    results_path = args.output / 'eval_results.json'
    results_path.write_text(
        json.dumps({'quality': quality, 'runtime': runtime}, indent=2),
        encoding='utf-8',
    )
    print(f'Full results saved to {results_path}')


if __name__ == '__main__':
    main()
