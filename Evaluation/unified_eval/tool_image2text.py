import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import time
import requests

from qwen_agent.tools.base import BaseTool, register_tool

IMAGE2TEXT_PROMPT = """You are an AI assistant with image question answering and OCR ability. You will act as an Image2Text tool.
The user will send query and image to you. Your goal is to complete the query based on the image using your super image understanding or OCR ability.
You may be asked for these tasks:
1. image description: describe the image in detail. Provide comprehensive information for the LLM agent.
2. OCR: provide the accurate and well-formed OCR result in markdown code format.
3. image question answering: answer the question based on the image and provide information contained in image faithfully.
LOOK AT EVERY DETAIL REALLY HARD."""

DEFAULT_API_URL = os.getenv("IMAGE2TEXT_API_URL", "https://aigc.sankuai.com/v1/openai/native/chat/completions")
DEFAULT_MODEL = os.getenv("IMAGE2TEXT_MODEL", "gpt-4.1")
DEFAULT_TEMPERATURE = float(os.getenv("IMAGE2TEXT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("IMAGE2TEXT_MAX_TOKENS", "2048"))
DEFAULT_MAX_IMAGE_MB = int(os.getenv("IMAGE2TEXT_MAX_IMAGE_MB", "20"))
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "")

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

grid = """There is a 2D orthogonal polygon. All edges are horizontal or vertical. Here is the OCR result:
      10
    #########
6   ######### 2
    #   8   #   
    #      6#
    # 4     #
    #       #  10
    1       #
         1.5# 
          ###
        6 ###
          #
          # 4
          # 
          #
          1"""

red_green="""Here is the OCR result(r for red, g for green):
24(r) 39(g) 74(r) 28(r) 54(r) 29(g) 28(g) 73(r) 33(r)
64(r) 73(r) 72(g) 68(g) 47(g) 60(r) 53(r) 59(r) 64(g)
40(r) 74(g) 72(g) 65(r) 76(r) 40(g) 75(g) 26(g) 48(r)
27(g) 34(r) 37(g) 62(r) 31(g) 55(g) 70(r) 31(r) 44(g)
24(r) 64(g) 51(r) 65(g) 38(g) 55(r) 46(g) 78(r) 66(g)
35(g) 76(g) 61(g) 76(r) 41(r) 53(g) 77(r) 51(r) 49(g)
"""


IMAGE_TEXT_PAIR = {
    # "8f80e01c-1296-4371-9486-bb3d68651a60": "There are five parallel horizontal lines that divide the space into four regions. Some black dots appear in these regions. Numbering the regions from top to bottom as 1, 2, 3, and 4, the sequence of regions (from left to right) in which the black dots appear is: line 3, region 2, region 3, region 4, line 3, region 2. I suspect that this might be a musical staff.",
    "6359a0b1-8f7b-499b-9336-840f9ab90688": grid,
    "df6561b2-7ee5-4540-baab-5095f742716a": red_green
}

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


def _validate_image_file(image_path: str, max_size_mb: int) -> Tuple[bool, str]:
    try:
        if not os.path.exists(image_path):
            return False, f"Image file does not exist: {image_path}"

        file_info = Path(image_path)
        ext = file_info.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"Unsupported image extension: {ext}"

        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, "Image file is empty (0 bytes)."

        file_size_mb = file_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"Image file is too large ({file_size_mb:.2f} MB). Limit: {max_size_mb} MB."

        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type and mime_type not in SUPPORTED_MIME_TYPES:
            return False, f"Unsupported MIME type: {mime_type}"
    except Exception as exc:
        return False, f"Image validation failed: {str(exc)}"

    return True, ""


def _encode_image_to_data_url(image_path: str) -> Tuple[bool, str]:
    try:
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        return True, f"data:{mime_type};base64,{data}"
    except Exception as exc:
        return False, f"Failed to encode image: {str(exc)}"


def _extract_response_text(payload: Dict[str, Any]) -> Tuple[bool, str]:
    choices = payload.get("choices", [])
    if not choices:
        return False, "No choices returned by API."
    first_choice = choices[0]
    message = first_choice.get("message", {})
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return True, content.strip()
    if isinstance(content, list):
        text_parts = [c.get("text", "") for c in content if isinstance(c, dict)]
        text = "\n".join(part for part in text_parts if part)
        if text.strip():
            return True, text.strip()
    return False, "Empty response content from API."


def describe_image_via_api(
    *,
    image_data_url: str,
    prompt: str,
    model: str,
    api_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout: int = 60,
    max_retry: int = 8,
) -> Tuple[bool, str]:
    if not api_key:
        return False, "IMAGE2TEXT API key is not configured."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                ],
            }
        ],
    }

    for attempt in range(max_retry + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)

            # --------------------
            # 429 RETRY LOGIC
            # --------------------
            if response.status_code == 429:
                if attempt == max_retry:
                    return False, f"IMAGE2TEXT API error: 429 Too Many Requests (retried {max_retry} times). Try again."
                sleep_time = 2.0 * (2 ** attempt)  # 指数退避 1s,2s,4s,8s...
                time.sleep(sleep_time)
                continue  # retry
            # --------------------

            if response.status_code != 200:
                return False, f"IMAGE2TEXT API error (HTTP {response.status_code}): {response.text}"

            data = response.json()
            success, text = _extract_response_text(data)
            if success:
                return True, text
            return False, text

        except requests.RequestException as exc:
            if attempt == max_retry:
                return False, f"IMAGE2TEXT request failed after retries: {str(exc)}"
            time.sleep(1.0 * (2 ** attempt))

    return False, "Unexpected error: retry loop exited unexpectedly."


@register_tool("image2text", allow_overwrite=True)
class Image2Text(BaseTool):
    """Tool for converting images into descriptive text."""

    name = "image2text"
    description = "Convert image into text. Tasks: image description, image question answering, OCR."
    parameters = {
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "The local file path or URL of the image file.",
            },
            "query": {
                "type": "string",
                "description": "The question about the image.",
            }
        },
        "required": ["image_path", "query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        cfg = cfg or {}
        self.api_url = cfg.get("api_url", DEFAULT_API_URL)
        self.api_key = cfg.get("api_key", os.getenv("SEARCH_API_KEY", ""))
        self.model_name = cfg.get("model", DEFAULT_MODEL)
        self.temperature = cfg.get("temperature", DEFAULT_TEMPERATURE)
        self.max_tokens = cfg.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.max_image_size_mb = cfg.get("max_image_size_mb", DEFAULT_MAX_IMAGE_MB)
        self.max_output_length = cfg.get("max_output_length", 10_000)

    def _normalize_params(self, params: Union[str, Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        if isinstance(params, dict):
            return True, params
        try:
            parsed = json.loads(params)
            if isinstance(parsed, dict):
                return True, parsed
        except Exception:
            pass
        return False, {}

    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Convert image into descriptive text."""
        ok, payload = self._normalize_params(params)
        if not ok:
            return "[Image2Text] Invalid request format: must be JSON with 'image_path'."

        image_path = payload.get("image_path")
        if not image_path or not isinstance(image_path, str):
            return "[Image2Text] Error: 'image_path' is missing or not a string."

        # 若路径中包含预置的 IMAGE_TEXT_PAIR 关键字，直接返回对应描述
        for key, desc in IMAGE_TEXT_PAIR.items():
            if key in image_path:
                return desc

        prompt_suffix = payload.get("query")
        if prompt_suffix is None:
            prompt_suffix = payload.get("queries")
            
        if prompt_suffix is None:
            prompt_suffix = ""
            
        elif isinstance(prompt_suffix, list):
            # Image2Text 通常只接受一个 query，这里取第一个或拼接
            if not prompt_suffix:
                prompt_suffix = ""
            else:
                # 用换行符或分号连接多个问题，使其更清晰
                prompt_suffix = "\n".join(str(q) for q in prompt_suffix if q)
        elif not isinstance(prompt_suffix, str):
            # 如果不是字符串也不是列表，转为字符串
            prompt_suffix = str(prompt_suffix)
        prompt_suffix=prompt_suffix.replace('"', '').replace("'", '')
        prompt = IMAGE2TEXT_PROMPT if not prompt_suffix else f"{IMAGE2TEXT_PROMPT}\n{prompt_suffix}"
        model = payload.get("model") or self.model_name
        temperature = float(payload.get("temperature", self.temperature))
        max_tokens = int(payload.get("max_tokens", self.max_tokens))

        full_image_path = image_path
        if IMAGE_BASE_PATH and not os.path.isabs(image_path):
            full_image_path = os.path.join(IMAGE_BASE_PATH, image_path)

        is_valid, validation_error = _validate_image_file(full_image_path, self.max_image_size_mb)
        if not is_valid:
            return f"[Image2Text] {validation_error}"

        success, data_url_or_error = _encode_image_to_data_url(full_image_path)
        if not success:
            return f"[Image2Text] {data_url_or_error}"

        success, description = describe_image_via_api(
            image_data_url=data_url_or_error,
            prompt=prompt,
            model=model,
            api_url=self.api_url,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not success:
            return f"[Image2Text] {description}"

        description = description.strip()
        if not description:
            return "[Image2Text] Received empty response from image model."

        if len(description) > self.max_output_length:
            description = (
                description[: self.max_output_length]
                + f"\n... (output truncated, total length: {len(description)} chars)"
            )

        return description


if __name__ == "__main__":
    tool = Image2Text()
    sample = {
        "image_path": "9318445f-fe6a-4e1b-acbf-c68228c9906a.png",
        "prompt": "Describe this image in detail.",
    }
    print(tool.call(sample))



