# -*- coding: utf-8 -*-
"""Validate Label Studio tasks/exports and compare predictions when references exist."""
from __future__ import annotations
import argparse
import json
import math
import re
from pathlib import Path


def result_boxes(result):
    text_by_id = {
        r.get("id"): "".join(r.get("value", {}).get("text", []))
        for r in result if r.get("type") == "textarea"
    }
    out = []
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        value = r.get("value", {})
        try:
            values = {k: float(value[k]) for k in ("x", "y", "width", "height")}
        except (KeyError, TypeError, ValueError):
            continue
        out.append({"id": r.get("id"), **values,
                    "label": (value.get("rectanglelabels") or [""])[0],
                    "text": text_by_id.get(r.get("id"), "")})
    return out


def all_annotations(task):
    anns = list(task.get("annotations", []))
    anns.extend(task.get("predictions", []))
    return anns


def validate_task(task):
    errors = []
    if not isinstance(task.get("data", {}).get("image"), str):
        errors.append("missing data.image")
    for annotation in all_annotations(task):
        result = annotation.get("result", [])
        rectangles = {r.get("id") for r in result if r.get("type") == "rectanglelabels"}
        for r in result:
            if r.get("type") == "textarea" and r.get("id") not in rectangles:
                errors.append("textarea is not linked to a rectangle region")
        for box in result_boxes(result):
            values = [box[k] for k in ("x", "y", "width", "height")]
            if box["label"] != "算式":
                errors.append("unexpected label")
            elif not all(math.isfinite(v) for v in values):
                errors.append("non-finite coordinate")
            elif min(box["x"], box["y"]) < 0 or min(box["width"], box["height"]) <= 0:
                errors.append("invalid coordinate or empty box")
            elif box["x"] + box["width"] > 100.001 or box["y"] + box["height"] > 100.001:
                errors.append("box outside image")
    return errors


def iou(a, b):
    iw = max(0, min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"]))
    ih = max(0, min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union > 0 else 0.0


def normalize(text):
    return re.sub(r"\s+", "", text).replace("×", "*").replace("÷", "/").replace("−", "-")


def compare(gt, pred):
    pairs = sorted((iou(p, g), pi, gi)
                   for pi, p in enumerate(pred)
                   for gi, g in enumerate(gt)
                   if p["label"] == g["label"])[::-1]
    used_p, used_g, scores = set(), set(), []
    text_total = text_correct = 0
    for score, pi, gi in pairs:
        if score < 0.5 or pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); scores.append(score)
        if gt[gi]["text"]:
            text_total += 1
            text_correct += normalize(gt[gi]["text"]) == normalize(pred[pi]["text"])
    return {"gt_boxes": len(gt), "pred_boxes": len(pred), "matched": len(scores),
            "iou_sum": sum(scores), "matched_reference_texts": text_total,
            "text_exact_matches": text_correct}


def task_key(task):
    return task.get("data", {}).get("filename") or str(task.get("id"))


def evaluate_predictions(tasks, prediction_tasks):
    by_key = {task_key(t): t for t in prediction_tasks}
    metrics = {"evaluated_images": 0, "images_without_reference": 0,
               "images_missing_prediction": 0, "gt_boxes": 0, "pred_boxes": 0,
               "matched": 0, "iou_sum": 0.0, "matched_reference_texts": 0,
               "text_exact_matches": 0}
    for task in tasks:
        refs = [a for a in task.get("annotations", []) if a.get("ground_truth")]
        if not refs:
            metrics["images_without_reference"] += 1
            continue
        pred_task = by_key.get(task_key(task))
        predictions = (pred_task or {}).get("predictions", [])
        if not predictions and pred_task and "result" in pred_task:
            predictions = [pred_task]
        if not predictions:
            metrics["images_missing_prediction"] += 1
            continue
        metrics["evaluated_images"] += 1
        for k, value in compare(result_boxes(refs[0].get("result", [])), result_boxes(predictions[0].get("result", []))).items():
            metrics[k] += value
    matched = metrics["matched"]
    metrics["precision_iou50"] = matched / metrics["pred_boxes"] if metrics["pred_boxes"] else None
    metrics["recall_iou50"] = matched / metrics["gt_boxes"] if metrics["gt_boxes"] else None
    metrics["mean_matched_iou"] = metrics.pop("iou_sum") / matched if matched else None
    metrics["text_exact_match_rate_on_matched_reference_texts"] = (metrics["text_exact_matches"] / metrics["matched_reference_texts"] if metrics["matched_reference_texts"] else None)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text(encoding="utf-8-sig"))
    errors, split_stats = [], {}
    for task in tasks:
        errors.extend(f"task {task.get('id')}: {message}" for message in validate_task(task))
        split = task.get("data", {}).get("split", "unknown")
        stats = split_stats.setdefault(split, {"images": 0, "boxes": 0, "texts": 0})
        stats["images"] += 1
        for annotation in task.get("annotations", []):
            boxes = result_boxes(annotation.get("result", []))
            stats["boxes"] += len(boxes)
            stats["texts"] += sum(bool(box["text"]) for box in boxes)
    report = {"tasks": len(tasks), "split_stats": split_stats, "validation_errors": errors}
    if args.predictions:
        prediction_tasks = json.loads(args.predictions.read_text(encoding="utf-8-sig"))
        report["prediction_metrics"] = evaluate_predictions(tasks, prediction_tasks)
        report["note"] = "Prediction metrics require ground_truth annotations; unlabelled test images are excluded."
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
