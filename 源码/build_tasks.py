"""Build Label Studio tasks from the supplied arithmetic worksheet dataset."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.parse import quote

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = (("train", True), ("同类测试", False), ("鲁棒测试", False))


def load_annotations(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["image"]).lower(): item for item in raw}


def result_for_box(box: dict, idx: int, width: int, height: int) -> list[dict]:
    region_id = f"r{idx:03d}"
    rectangle = {
        "id": region_id,
        "from_name": "formula_box",
        "to_name": "image",
        "type": "rectanglelabels",
        "value": {
            "x": float(box["x"]), "y": float(box["y"]),
            "width": float(box["width"]), "height": float(box["height"]),
            "rotation": float(box.get("rotation", 0)),
            "rectanglelabels": [str(box.get("label", "算式"))],
        },
        "original_width": width,
        "original_height": height,
        "image_rotation": 0,
    }
    text = str(box.get("text", "")).strip()
    if not text:
        return [rectangle]
    textarea = {
        "id": region_id, "from_name": "formula_text", "to_name": "image",
        "type": "textarea", "value": {"text": [text]},
        "original_width": width, "original_height": height,
        "image_rotation": 0,
    }
    return [rectangle, textarea]


def make_task(image_path: Path, data_root: Path, prefix: str, ann: dict | None, split: str, task_id: int) -> dict:
    rel = image_path.relative_to(data_root.parent).as_posix()
    # The prefix is configurable because Cloud Studio and local Label Studio may differ.
    image_url = prefix + quote(rel, safe="/")
    width = int((ann or {}).get("original_width", 0))
    height = int((ann or {}).get("original_height", 0))
    if not width or not height:
        from PIL import Image
        with Image.open(image_path) as im:
            width, height = im.size
    boxes = list((ann or {}).get("boxes", []))
    results = []
    for idx, box in enumerate(boxes, 1):
        results.extend(result_for_box(box, idx, width, height))
    task = {
        "id": task_id,
        "data": {"image": image_url, "split": split, "filename": rel},
        "meta": {"source": "实验2数据", "split": split, "ground_truth_available": bool(ann)},
    }
    if ann:
        task["annotations"] = [{
            "ground_truth": True, "result": results,
        }]
    return task


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-url-prefix", default="/data/local-files/?d=")
    args = ap.parse_args()
    root = args.data_root.resolve()
    tasks: list[dict] = []
    task_id = 1
    counts = {}
    for split, has_ann in SPLITS:
        image_dir = root / split / "images"
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        ann_map = load_annotations(root / split / "annotations.json") if has_ann else {}
        images = sorted((p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
        counts[split] = len(images)
        for image in images:
            ann = ann_map.get(image.name.lower())
            if has_ann and ann is None:
                raise ValueError(f"Missing supplied annotation: {image.name}")
            tasks.append(make_task(image, root, args.image_url_prefix, ann, split, task_id))
            task_id += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    total_boxes = sum(sum(r.get("type") == "rectanglelabels" for r in t.get("annotations", [{}])[0].get("result", [])) for t in tasks if t.get("annotations"))
    print(json.dumps({"output": str(args.output), "tasks": len(tasks), "images_by_split": counts, "train_boxes": total_boxes}, ensure_ascii=False))


if __name__ == "__main__":
    main()
