"""Import tasks using a token from the environment; never serialize credentials."""
from pathlib import Path
import argparse
import json
import os
import sys

import requests

ROOT = Path(__file__).resolve().parent
TITLE = "实验2 数据标注实践"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("LABEL_STUDIO_URL") or
                        "http://127.0.0.1:" + os.getenv("APP_PORT", "8080"))
    parser.add_argument("--tasks", type=Path, default=ROOT / ".runtime/tasks.json")
    parser.add_argument("--backend-url", default="http://127.0.0.1:" + os.getenv("ML_BACKEND_PORT", "9090"))
    parser.add_argument("--connect-backend", action="store_true")
    args = parser.parse_args()
    token = os.getenv("LABEL_STUDIO_API_KEY", "").strip()
    if not token:
        parser.error("Set LABEL_STUDIO_API_KEY from Account & Settings before importing.")
    tasks = json.loads(args.tasks.read_text(encoding="utf-8-sig"))
    base = args.url.rstrip("/")
    session = requests.Session()
    session.headers["Authorization"] = "Token " + token

    def request(method, path, **kwargs):
        response = session.request(method, base + path, timeout=120, **kwargs)
        # Do not echo server error bodies or request headers containing credentials.
        if not response.ok:
            raise RuntimeError(f"Label Studio {method} {path} returned HTTP {response.status_code}.")
        return response.json()

    def project_list():
        page, rows = 1, []
        while True:
            batch = request("GET", f"/api/projects/?page={page}&page_size=100")
            items = batch.get("results", []) if isinstance(batch, dict) else batch
            rows.extend(items)
            if not isinstance(batch, dict) or not batch.get("next"):
                return rows
            page += 1

    config = (ROOT / "配置/label_config.xml").read_text(encoding="utf-8-sig")
    project = next((p for p in project_list() if p["title"] == TITLE), None)
    if project is None:
        project = request("POST", "/api/projects/", json={"title": TITLE, "label_config": config,
                          "description": "训练集参考标注；两个测试集等待独立预标注和人工复核。"})
    pid = project["id"]
    storage_path = str((ROOT / "实验2数据").resolve())
    storages = request("GET", f"/api/storages/localfiles/?project={pid}")
    if not any(item["path"] == storage_path for item in storages):
        request("POST", "/api/storages/localfiles/", json={"project": pid, "path": storage_path,
                "use_blob_urls": True, "title": "实验2课程图片（任务 JSON 单独导入）"})

    def task_list():
        rows, page = [], 1
        while True:
            batch = request("GET", f"/api/tasks/?project={pid}&page_size=100&page={page}")
            items = batch.get("tasks", batch.get("results", [])) if isinstance(batch, dict) else batch
            rows.extend(items)
            if len(items) < 100:
                return rows
            page += 1

    seen = {t["data"].get("filename") for t in task_list()}
    pending = [{k: v for k, v in t.items() if k != "id"}
               for t in tasks if t["data"]["filename"] not in seen]
    if pending:
        request("POST", f"/api/projects/{pid}/import", json=pending)
    if args.connect_backend:
        backends = request("GET", f"/api/ml/?project={pid}")
        if isinstance(backends, dict):
            backends = backends.get("results", [])
        if not any(b["url"].rstrip("/") == args.backend_url.rstrip("/") for b in backends):
            request("POST", "/api/ml/", json={"project": pid, "url": args.backend_url,
                    "title": "Qwen VL + OCR", "timeout": 3600})
    rows = task_list()
    if rows:
        image = session.get(base + rows[0]["data"]["image"], timeout=30)
        if image.status_code != 200 or not image.headers.get("Content-Type", "").startswith("image/"):
            raise RuntimeError("Imported image is not accessible; check local-files storage authorization.")
    summary = {"project_id": pid, "task_count": len(rows), "new_tasks": len(pending),
               "project_path": f"/projects/{pid}/data"}
    state = args.tasks.parent
    (state / "import_result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
