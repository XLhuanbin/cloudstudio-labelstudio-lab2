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
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

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
    """Native synchronous PaddleOCR-VL API from AI Studio's API example."""

    def chat(self, image: bytes, prompt: str) -> str:
        if not self.url:
            return ""
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
        texts = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            # Prefer raw block content, preserving the model's symbols and answers.
            pruned = page.get("prunedResult", {})
            blocks = pruned.get("parsing_res_list", []) if isinstance(pruned, dict) else []
            block_texts = [block["block_content"].strip() for block in blocks
                           if isinstance(block, dict) and isinstance(block.get("block_content"), str)
                           and block["block_content"].strip()] if isinstance(blocks, list) else []
            if block_texts:
                texts.extend(block_texts)
            else:
                markdown = page.get("markdown", {})
                text = markdown.get("text", "") if isinstance(markdown, dict) else ""
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
        if not texts:
            raise RuntimeError("PaddleOCR-VL returned no recognized text for this crop")
        return "\n".join(texts)


class NewModel(LabelStudioMLBase):
    """Cloud Studio ML backend entry point."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vision = Provider("QWEN_VL_API_URL", "QWEN_VL_API_KEY")
        style = os.getenv("OCR_API_STYLE", "openai").strip().lower()
        if style not in {"openai", "paddleocr"}:
            raise ValueError("OCR_API_STYLE must be openai or paddleocr")
        provider = PaddleOCRProvider if style == "paddleocr" else Provider
        self.ocr = provider("OCR_API_URL", "OCR_API_KEY", "OCR_MODEL", "HunyuanOCR")

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
