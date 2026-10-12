"""Label Studio ML backend for Qwen-VL box proposals plus OCR refinement.

The backend is intentionally endpoint-agnostic: Cloud Studio can point the two
providers at Qwen-VL and PaddleOCR-VL/HunyuanOCR compatible HTTP endpoints.
"""
from __future__ import annotations
import base64
import json
import os
import re
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from PIL import Image
from io import BytesIO

from label_studio_ml.model import LabelStudioMLBase


PROMPT = """你是算术练习册标注器。请识别图片中每一个算式所在的矩形区域，并读取算式文本。
只输出 JSON 数组，不要 Markdown。每项格式为
{\"x\":数字,\"y\":数字,\"width\":数字,\"height\":数字,\"text\":字符串,\"score\":数字}。
坐标必须是相对于原图左上角的百分比，范围 0 到 100；框应覆盖印刷算式和手写答案。"""


def extract_json(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start = min([p for p in (text.find("["), text.find("{")) if p >= 0], default=-1)
    if start >= 0:
        text = text[start:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return []


def image_bytes(value: str) -> bytes:
    if value.startswith("data:"):
        return base64.b64decode(value.split(",", 1)[1])
    parsed = urlparse(value)
    if parsed.path == "/data/local-files/":
        rel = unquote(parse_qs(parsed.query).get("d", [""])[0])
        root = Path(os.getenv("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", ".")).resolve()
        target = (root / rel).resolve()
        if root not in target.parents:
            raise ValueError("Image must be inside the configured document root")
        return target.read_bytes()
    if parsed.scheme in {"http", "https"}:
        headers = {}
        ls_url = os.getenv("LABEL_STUDIO_URL", "")
        if ls_url and urlparse(ls_url).netloc == parsed.netloc:
            headers["Authorization"] = "Token " + os.getenv("LABEL_STUDIO_API_KEY", "")
        r = requests.get(value, headers=headers, timeout=60); r.raise_for_status(); return r.content
    return Path(value).read_bytes()


def data_uri(raw: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


def paddle_jobs_url(url: str) -> bool:
    return urlparse(url).path.rstrip("/") == "/api/v2/ocr/jobs"


class Provider:
    def __init__(self, url_var: str, key_var: str = "", model_var: str = "QWEN_VL_MODEL", default_model: str = "Qwen2.5-VL-72B-Instruct"):
        self.url = os.getenv(url_var, "").strip()
        self.key = os.getenv(key_var, "").strip() if key_var else ""
        self.model = os.getenv(model_var, default_model)

    def chat(self, image: bytes, prompt: str) -> Any:
        if not self.url:
            return []
        headers = {"Content-Type": "application/json"}
        if self.key: headers["Authorization"] = "Bearer " + self.key
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_uri(image)}}]}]}
        response = requests.post(self.url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", body)
        if isinstance(content, list): content = "\n".join(str(x.get("text", x)) for x in content)
        return content


class PaddleOCRProvider(Provider):
    """PaddleOCR-VL provider for both legacy sync and current async APIs."""

    def chat(self, image: bytes, prompt: str) -> str:
        if not self.url:
            return ""
        if paddle_jobs_url(self.url):
            return self._chat_v2(image)
        return self._chat_legacy(image)

    @staticmethod
    def _text_from_result(result: Any) -> str:
        """Extract readable text from one PaddleOCR-VL JSON result."""
        pages = result.get("layoutParsingResults") if isinstance(result, dict) else None
        if not isinstance(pages, list):
            return ""
        texts = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            pruned = page.get("prunedResult", {})
            blocks = pruned.get("parsing_res_list", []) if isinstance(pruned, dict) else []
            block_texts = [block["block_content"].strip() for block in blocks
                           if isinstance(block, dict) and isinstance(block.get("block_content"), str)
                           and block["block_content"].strip()] if isinstance(blocks, list) else []
            if block_texts:
                texts.extend(block_texts)
                continue
            markdown = page.get("markdown", {})
            text = markdown.get("text", "") if isinstance(markdown, dict) else ""
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())
        return "\n".join(texts)

    def _chat_legacy(self, image: bytes) -> str:
        """Legacy synchronous endpoint returned result.layoutParsingResults."""
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "token " + self.key
        payload = {
            "file": base64.b64encode(image).decode("ascii"),
            "fileType": 1,
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useChartRecognition": False,
            "useLayoutDetection": False,
            "promptLabel": "ocr",
            "prettifyMarkdown": False,
        }
        response = requests.post(self.url, headers=headers, json=payload, timeout=180)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError("PaddleOCR-VL returned an invalid response object")
        if body.get("errorCode", 0) != 0:
            raise RuntimeError("PaddleOCR-VL API error: " + str(body.get("errorMsg", body["errorCode"])))
        result = body.get("result")
        pages = result.get("layoutParsingResults") if isinstance(result, dict) else None
        if not isinstance(pages, list):
            raise RuntimeError("PaddleOCR-VL response is missing result.layoutParsingResults; use the synchronous API URL")
        text = self._text_from_result(result)
        if not text:
            raise RuntimeError("PaddleOCR-VL returned no recognized text")
        return text

    def _chat_v2(self, image: bytes) -> str:
        """Current AI Studio API: multipart submit, poll, then download JSONL."""
        if not self.model.startswith("PaddleOCR-VL"):
            raise RuntimeError("Set OCR_MODEL to PaddleOCR-VL-1.6 (or your selected PaddleOCR-VL version) for the jobs API")
        if not self.key:
            raise RuntimeError("OCR_API_KEY is not configured for the PaddleOCR jobs API")
        headers = {"Authorization": "Bearer " + self.key}
        options = {
            "useDocOrientationClassify": False,
            "useDocUnwarping": False,
            "useLayoutDetection": False,
            "promptLabel": "ocr",
            "useChartRecognition": False,
            "prettifyMarkdown": False,
        }
        data = {"model": self.model,
                "optionalPayload": json.dumps(options, ensure_ascii=False)}
        files = {"file": ("crop.jpg", image, "image/jpeg")}
        jobs_url = self.url.rstrip("/")
        response = requests.post(jobs_url, headers=headers, data=data, files=files, timeout=180)
        job_id = self._job_data(response, "submit").get("jobId")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError("PaddleOCR-VL submit response has no data.jobId")
        print(f"PaddleOCR submitted {job_id}; model={self.model}", flush=True)
        status_url = jobs_url + "/" + quote(job_id, safe="")
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            status_response = requests.get(status_url, headers=headers, timeout=min(60, remaining))
            job = self._job_data(status_response, "status")
            state = job.get("state")
            if state == "done":
                urls = job.get("resultUrl", {}) or {}
                json_url = urls.get("jsonUrl") if isinstance(urls, dict) else None
                if not json_url:
                    raise RuntimeError("PaddleOCR-VL completed without data.resultUrl.jsonUrl")
                result_response = requests.get(json_url, timeout=120)
                if not result_response.ok:
                    raise RuntimeError(f"PaddleOCR-VL result download failed with HTTP {result_response.status_code}")
                texts = []
                for line in result_response.text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        raise RuntimeError("PaddleOCR-VL returned invalid JSONL") from None
                    if not isinstance(record, dict):
                        raise RuntimeError("PaddleOCR-VL JSONL record must be an object")
                    text = self._text_from_result(record.get("result", record))
                    if text:
                        texts.append(text)
                if not texts:
                    raise RuntimeError("PaddleOCR-VL returned no recognized text")
                print(f"PaddleOCR completed {job_id}", flush=True)
                return "\n".join(texts)
            if state == "failed":
                message = str(job.get("errorMsg", "unknown error")).replace(self.key, "[redacted]")
                raise RuntimeError("PaddleOCR-VL job failed: " + message[:300])
            if state not in {"pending", "running"}:
                raise RuntimeError("PaddleOCR-VL returned an invalid job state")
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(3, remaining))
        raise RuntimeError("PaddleOCR-VL job timed out after 180 seconds")

    @staticmethod
    def _job_data(response: requests.Response, action: str) -> dict:
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError(f"PaddleOCR-VL {action} returned HTTP {response.status_code} without JSON") from None
        if not isinstance(body, dict):
            raise RuntimeError(f"PaddleOCR-VL {action} response must be a JSON object")
        code = body.get("code", 0)
        if not response.ok or code != 0:
            # Return useful error codes without echoing uploaded content or credentials.
            reasons = {10001: "empty file", 10004: "unsupported file format",
                       10007: "invalid OCR_MODEL", 10008: "invalid optionalPayload",
                       12001: "daily quota exceeded", 12002: "requests too frequent"}
            reason = reasons.get(code, "check API token, model and account quota") if isinstance(code, int) else "invalid API response"
            raise RuntimeError(f"PaddleOCR-VL {action} failed: HTTP {response.status_code}, code={code if isinstance(code, int) else 'invalid'}; {reason}")
        data = body.get("data")
        if not isinstance(data, dict):
            raise RuntimeError(f"PaddleOCR-VL {action} response has no data object")
        return data


class NewModel(LabelStudioMLBase):
    """Cloud Studio ML backend entry point."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision = Provider("QWEN_VL_API_URL", "QWEN_VL_API_KEY")
        style = os.getenv("OCR_API_STYLE", "openai").strip().lower()
        if style not in {"openai", "paddleocr"}:
            raise ValueError("OCR_API_STYLE must be openai or paddleocr")
        is_jobs = paddle_jobs_url(os.getenv("OCR_API_URL", "").strip())
        provider = PaddleOCRProvider if style == "paddleocr" or is_jobs else Provider
        default_model = "PaddleOCR-VL-1.6" if is_jobs else "HunyuanOCR"
        self.ocr = provider("OCR_API_URL", "OCR_API_KEY", "OCR_MODEL", default_model)

    def _prediction(self, task: dict) -> dict:
        value = task.get("data", {}).get("image", "")
        raw = image_bytes(value)
        with Image.open(BytesIO(raw)) as im:
            image = im.convert("RGB")
        width, height = image.size
        if not self.vision.url:
            raise RuntimeError("QWEN_VL_API_URL is not configured; no model inference has been run")
        proposals = extract_json(self.vision.chat(raw, PROMPT))
        if isinstance(proposals, dict): proposals = proposals.get("boxes", proposals.get("items", []))
        if not isinstance(proposals, list): proposals = []
        print(f"Task {task.get('id', '?')}: received {len(proposals)} vision proposals", flush=True)
        results = []
        for idx, item in enumerate(proposals, 1):
            if not isinstance(item, dict): continue
            try:
                x, y = float(item["x"]), float(item["y"])
                w, h = float(item["width"]), float(item["height"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(n) for n in (x, y, w, h)): continue
            if min(x, y) < 0 or min(w, h) <= 0 or x + w > 100 or y + h > 100: continue
            rid = f"p{idx:03d}"
            score = max(0.0, min(1.0, float(item.get("score", 0.5))))
            results.append({"id": rid, "from_name": "formula_box", "to_name": "image", "type": "rectanglelabels", "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "rectanglelabels": ["算式"]}, "score": score, "original_width": width, "original_height": height, "image_rotation": 0})
            text = str(item.get("text", "")).strip()
            if self.ocr.url:
                print(f"Task {task.get('id', '?')}: OCR crop {idx}/{len(proposals)}", flush=True)
                text = self._ocr_crop(image, x, y, w, h) or text
            if text:
                results.append({"id": rid, "from_name": "formula_text", "to_name": "image", "type": "textarea", "value": {"text": [text]}, "score": score, "original_width": width, "original_height": height, "image_rotation": 0})
        return {"result": results, "model_version": "qwen-vl-plus-ocr"}

    def _ocr_crop(self, image: Image.Image, x: float, y: float, w: float, h: float) -> str:
        left, top = int(x / 100 * image.width), int(y / 100 * image.height)
        right, bottom = int((x + w) / 100 * image.width), int((y + h) / 100 * image.height)
        crop = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
        buf = BytesIO(); crop.save(buf, format="JPEG", quality=95)
        result = self.ocr.chat(buf.getvalue(), "请逐字读取这个算式，包括手写答案，不要计算或修正答案。只输出算式文本。")
        if isinstance(result, str) and result.strip().startswith(("{", "[", "```")):
            result = extract_json(result)
        if isinstance(result, dict): return str(result.get("text", "")).strip()
        if isinstance(result, list) and result: return str(result[0].get("text", result[0])).strip() if isinstance(result[0], dict) else str(result[0]).strip()
        return str(result).strip() if isinstance(result, str) else ""

    def predict(self, tasks, **kwargs):
        return [self._prediction(task) for task in tasks]

    def fit(self, tasks, workdir=None, **kwargs):
        # This is a pre-annotation backend; active training is intentionally optional.
        return {"status": "ready", "message": "No online fine-tuning configured."}
