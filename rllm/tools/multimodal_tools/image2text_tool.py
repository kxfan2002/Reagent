import base64
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Tuple

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

# Default configuration
# REFERENCE_COUNT = 8
DEFAULT_TIMEOUT = 60
IMAGE2TEXT_ENDPOINT = os.getenv("IMAGE2TEXT_API_URL", "https://xxxx.com/v1/openai/native/chat/completions")
DEFAULT_MODEL = os.getenv("IMAGE2TEXT_MODEL", "gpt-4.1")
DEFAULT_TEMPERATURE = float(os.getenv("IMAGE2TEXT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("IMAGE2TEXT_MAX_TOKENS", "2048"))
DEFAULT_MAX_IMAGE_MB = int(os.getenv("IMAGE2TEXT_MAX_IMAGE_MB", "20"))
IMAGE_BASE_PATH = os.getenv("IMAGE_BASE_PATH", "./data")

IMAGE2TEXT_PROMPT = """You are an AI assistant with image question answering and OCR ability. You will act as an Image2Text tool.
The user will send query and image to you. Your goal is to complete the query based on the image using your super image understanding or OCR ability.
You may be asked for these tasks:
1. image description: describe the image in detail. Provide comprehensive information for the LLM agent.
2. OCR: provide the accurate and well-formed OCR result in markdown code format.
3. image question answering: answer the question based on the image and provide information contained in image faithfully.
LOOK AT EVERY DETAIL REALLY HARD. Answer the question faithfully."""

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

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/tiff",
}


def _validate_image_file(image_path: str, max_size_mb: int) -> Tuple[bool, str]:
    """Validate image file existence, extension, and size."""
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
    """Encode image file to base64 data URL."""
    try:
        with open(image_path, "rb") as image_file:
            data = base64.b64encode(image_file.read()).decode("utf-8")
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        return True, f"data:{mime_type};base64,{data}"
    except Exception as exc:
        return False, f"Failed to encode image: {str(exc)}"


def _extract_response_text(payload: dict[str, Any]) -> Tuple[bool, str]:
    """Extract text content from API response."""
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


class Image2TextTool(Tool):
    """A tool for converting images into descriptive text using vision API."""

    NAME = "image2text"
    DESCRIPTION = "Convert image into text. Tasks: image description, image question answering, OCR."

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        api_url: str = IMAGE2TEXT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_image_size_mb: int = DEFAULT_MAX_IMAGE_MB,
        timeout: float = DEFAULT_TIMEOUT,
        max_retry: int = 20,
        max_output_length: int = 10000,
    ):
        """
        Initialize the Image2Text tool.

        Args:
            name (str): The name of the tool, defaults to Image2TextTool.NAME.
            description (str): A description of the tool's purpose.
            api_url (str): API endpoint URL for image processing.
            model (str): Model name to use for image processing.
            temperature (float): Temperature parameter for model inference.
            max_tokens (int): Maximum tokens to generate in response.
            max_image_size_mb (int): Maximum allowed image file size in MB.
            timeout (float): Maximum time in seconds to wait for API response.
            max_retry (int): Maximum number of retry attempts, defaults to 8.
            max_output_length (int): Maximum length of output text before truncation.
        """
        self.api_url = api_url
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size_mb = max_image_size_mb
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_output_length = max_output_length
        self._init_client()
        super().__init__(name=name, description=description)

    def _init_client(self):
        """Initialize the HTTP client for making requests."""
        self.client = httpx.Client()

    @property
    def json(self):
        """Return the tool's information in the required format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Convert image into text. Tasks: image description, image question answering, OCR.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "The local file path or URL of the image file."
                        },
                        "query": {
                            "type": "string",
                            "description": "The question about the image."
                        }
                    },
                    "required": ["image_path", "query"]
                }
            }
        }

    def _describe_image_via_api(
        self,
        image_data_url: str,
        prompt: str,
    ) -> Tuple[bool, str]:
        """
        Call the vision API to describe the image.

        Args:
            image_data_url (str): Base64-encoded image data URL
            prompt (str): Text prompt for the API

        Returns:
            Tuple[bool, str]: Success flag and result text or error message
        """
        api_key = os.getenv("SEARCH_API_KEY","")
        if not api_key:
            return False, "SEARCH_API_KEY environment variable is not set"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
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

        for attempt in range(self.max_retry):
            try:
                response = self.client.post(
                    url=self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )

                # Handle 429 Too Many Requests with exponential backoff
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = 2.0
                    else:
                        wait_time = 2.0 * (2 ** attempt)  # Exponential backoff

                    # Ensure minimum wait time of 2 seconds
                    wait_time = max(wait_time, 2.0)
                    # Add random jitter
                    wait_time += random.uniform(0, 2)

                    print(f"[Image2Text] 429 Too Many Requests. Waiting {wait_time:.1f}s before retry ({attempt + 1}/{self.max_retry})...", flush=True)
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

                data = response.json()
                success, text = _extract_response_text(data)
                if success:
                    return True, text
                return False, text

            except httpx.HTTPStatusError as e:
                print(f"[Image2Text] HTTP error on attempt {attempt + 1}/{self.max_retry}: {e}", flush=True)
                if attempt == self.max_retry - 1:
                    return False, f"IMAGE2TEXT API error (HTTP {e.response.status_code}): {e.response.text}"
                time.sleep(1.0 * (2 ** attempt))
            except Exception as e:
                print(f"[Image2Text] Request failed on attempt {attempt + 1}/{self.max_retry}: {e}", flush=True)
                if attempt == self.max_retry - 1:
                    return False, f"IMAGE2TEXT request failed after retries: {str(e)}"
                time.sleep(1.0 * (2 ** attempt))

        return False, "Unexpected error: retry loop exited unexpectedly."

    def forward(self, image_path: str, query: str) -> ToolOutput:
        """
        Convert image into descriptive text.

        Args:
            image_path (str): The local file path of the image.
            query (str): The question or instruction about the image.

        Returns:
            ToolOutput: An object containing either the description or an error message.
        """
        try:
            assert self.client is not None, "Image2Text Client not initialized"

            if not isinstance(image_path, str) or not image_path:
                return ToolOutput(
                    name=self.name or "image2text",
                    error="'image_path' must be a non-empty string"
                )

            if not isinstance(query, str):
                return ToolOutput(
                    name=self.name or "image2text",
                    error="'query' must be a string"
                )

            # Construct full prompt
            prompt = IMAGE2TEXT_PROMPT if not query else f"{IMAGE2TEXT_PROMPT}\n\n{query}"

            # Handle relative vs absolute paths
            full_image_path = image_path
            if IMAGE_BASE_PATH and not os.path.isabs(image_path):
                full_image_path = os.path.join(IMAGE_BASE_PATH, image_path)

            # Validate image file
            is_valid, validation_error = _validate_image_file(full_image_path, self.max_image_size_mb)
            if not is_valid:
                return ToolOutput(
                    name=self.name or "image2text",
                    error=validation_error
                )

            # Encode image to base64
            success, data_url_or_error = _encode_image_to_data_url(full_image_path)
            if not success:
                return ToolOutput(
                    name=self.name or "image2text",
                    error=data_url_or_error
                )

            # Call API
            success, description = self._describe_image_via_api(
                image_data_url=data_url_or_error,
                prompt=prompt,
            )
            if not success:
                return ToolOutput(
                    name=self.name or "image2text",
                    error=description
                )

            description = description.strip()
            if not description:
                return ToolOutput(
                    name=self.name or "image2text",
                    error="Received empty response from image model"
                )

            # Truncate if too long
            if len(description) > self.max_output_length:
                description = (
                    description[: self.max_output_length]
                    + f"\n... (output truncated, total length: {len(description)} chars)"
                )

            return ToolOutput(name=self.name or "image2text", output=description)

        except Exception as e:
            error_msg = f"Image2Text processing failed: {str(e)}"
            return ToolOutput(name=self.name or "image2text", error=error_msg)

    def __del__(self):
        """Clean up the HTTP client on deletion."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Test the tool
    tool = Image2TextTool()
    print("Testing Image2Text Tool...")
    print("=" * 80)

    # Test 1: Basic image description
    print("\nTest 1: Image description")
    result = tool(image_path="rl/images/OCR_VQA/000787376X.jpg", query="Describe this image in detail.")
    print(result)
    print("=" * 80)

