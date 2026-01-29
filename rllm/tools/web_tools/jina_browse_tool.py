import os
import re
import time
import random
from typing import Any, Optional

import httpx
try:
    import tiktoken
except ImportError:
    tiktoken = None

from rllm.tools.tool_base import Tool, ToolOutput

DEFAULT_TIMEOUT = 80
DEFAULT_MAX_RETRY = 100
JINA_ENDPOINT = "https://r.jina.ai/"


def clean_html_tags(text: str) -> str:
    """Remove HTML tags like <b> and </b>"""
    return re.sub(r'</?b>', '', text)


class JinaBrowseTool(Tool):
    """A tool for browsing webpages using Jina Reader API and extracting relevant information with LLM."""

    NAME = "browse"
    DESCRIPTION = "Browse a webpage and extract relevant information based on a specific query."

    def __init__(
        self, 
        name: str = NAME, 
        description: str = DESCRIPTION, 
        timeout: float = DEFAULT_TIMEOUT,
        max_retry: int = DEFAULT_MAX_RETRY,
        generate_engine: str = "deepseek",
        max_tokens_per_chunk: int = 95000
    ):
        """
        Initialize the JinaBrowse tool.

        Args:
            name (str): The name of the tool.
            description (str): A description of the tool's purpose.
            timeout (float): Maximum time in seconds to wait for Jina API response.
            max_retry (int): Maximum number of retry attempts for Jina API.
            generate_engine (str): LLM engine to use for content extraction. Options: "deepseek", "gemini", "openai".
            max_tokens_per_chunk (int): Maximum tokens per chunk when splitting long content.
        """
        self.timeout = timeout
        self.max_retry = max_retry
        self.generate_engine = generate_engine
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self._init_client()
        super().__init__(name=name, description=description)

    def _init_client(self):
        """Initialize the HTTP client and proxy settings."""
        proxy = os.getenv("HTTP_PROXY", "http://10.70.16.106:3128")
        # httpx uses 'proxy' or 'proxies' with correct URL format
        # For httpx >= 0.23.0, use 'proxy' parameter with a single proxy URL
        # or set environment variable HTTP_PROXY/HTTPS_PROXY
        try:
            # Try modern httpx API (0.23.0+)
            self.client = httpx.Client(proxy=proxy)
        except TypeError:
            # Fallback to environment variables or no proxy
            self.client = httpx.Client()

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Browse a webpage and extract relevant information based on a specific query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL of the webpage to browse."
                        },
                        "query": {
                            "type": "string",
                            "description": "The specific query to extract relevant information from the webpage."
                        }
                    },
                    "required": ["url", "query"]
                }
            }
        }

    def _read_page_with_jina(self, url: str) -> str:
        """
        Read webpage content using Jina Reader API.
        
        Args:
            url (str): The URL to read
            
        Returns:
            str: The webpage content as text
        """
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError("JINA_API_KEY environment variable is not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        for attempt in range(self.max_retry):
            try:
                response = self.client.get(
                    f"{JINA_ENDPOINT}{url}",
                    headers=headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = 4.0
                    else:
                        wait_time = 4.0
                    
                    wait_time = max(wait_time, 3.0)
                    wait_time += random.uniform(4, 9)
                    
                    print(f"[JinaBrowse] 429 Too Many Requests. Retrying in {wait_time:.1f} seconds (attempt {attempt + 1}/{self.max_retry})...", flush=True)
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 422 or response.status_code == 403:
                    print(f"[JinaBrowse] 422/403 Unprocessable Entity. {response.text}", flush=True)
                    raise Exception(f"Browse API 422/403 Unprocessable Entity: {response.text}\nDo not retry this URL.")
                elif response.status_code == 400:
                    print(f"[JinaBrowse] 400 Bad Request. {response.text}", flush=True)
                    raise Exception(f"Browse API 400 Bad Request: {response.text}\nThis should not happen. Did you use browse to read a file?(If so, use file_reader.)")
                else:
                    raise Exception(f"Browse API error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"[JinaBrowse] Attempt {attempt + 1}/{self.max_retry} failed: {e}", flush=True)
                error_msg = str(e)
                if "422/403 Unprocessable Entity" in error_msg or "400 Bad Request" in error_msg:
                    raise
                if attempt == self.max_retry - 1:
                    raise
                time.sleep(0.5 + random.uniform(0, 1))
        
        raise Exception(f"Failed to read page after {self.max_retry} attempts")

    def _get_llm_response(self, query: str, temperature: float = 0.0, max_retry: int = 10, 
                        fallback_engines: list[str] | None = None) -> Optional[str]:
        """
        Get response from LLM for content extraction.
        
        Args:
            query (str): The prompt to send to the LLM
            temperature (float): Sampling temperature
            max_retry (int): Maximum retry attempts per engine
            fallback_engines (list[str]): List of fallback engines to try if primary fails with 450
            
        Returns:
            Optional[str]: The LLM response or None if failed
        """
        try:
            from openai import OpenAI, RateLimitError
        except ImportError:
            raise ImportError("openai package is required. Install it with: pip install openai")

        if fallback_engines is None:
            engines_to_try = [self.generate_engine, "gemini", "openai"]
        else:
            engines_to_try = [self.generate_engine] + fallback_engines
        
        # 去重，保持顺序
        seen = set()
        engines_to_try = [x for x in engines_to_try if not (x in seen or seen.add(x))]
        
        last_error = None
        
        for engine_idx, current_engine in enumerate(engines_to_try):
            # 获取当前引擎配置
            if current_engine == "deepseek":
                api_key = os.getenv("DEEPSEEK_API_KEY")
                api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
                model_name = "deepseek-chat"
                max_tokens = 8192
            elif current_engine == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                api_base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
                model_name = "gemini-2.5-flash"
                max_tokens = 4096
            elif current_engine == "openai":
                api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
                api_base = os.getenv("OPENAI_API_BASE") or os.getenv("API_BASE")
                model_name = os.getenv("SUMMARY_MODEL_NAME", "gpt-4o-mini")
                max_tokens = 4096
            else:
                print(f"[JinaBrowse] Unknown engine: {current_engine}, skipping", flush=True)
                continue

            if not api_key:
                print(f"[JinaBrowse] {current_engine.upper()}_API_KEY not set, skipping this engine", flush=True)
                continue

            print(f"[JinaBrowse] Using LLM engine: {current_engine}", flush=True)
            client = OpenAI(api_key=api_key, base_url=api_base)

            for attempt in range(max_retry):
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": query}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    content = response.choices[0].message.content
                    if content:
                        if engine_idx > 0:
                            print(f"[JinaBrowse] Successfully used fallback engine: {current_engine}", flush=True)
                        return content
                    
                except RateLimitError as e:
                    # 专门处理 429 Rate Limit 错误
                    wait_time = random.uniform(1, 4)
                    print(f"[JinaBrowse] {current_engine} LLM 429 Rate Limit. Retrying in {wait_time:.1f} seconds (attempt {attempt + 1}/{max_retry})...", flush=True)
                    if attempt == max_retry - 1:
                        print(f"[JinaBrowse] {current_engine} Rate Limit exceeded after {max_retry} attempts", flush=True)
                        last_error = str(e)
                        break  
                    time.sleep(wait_time)
                    
                except Exception as e:
                    error_str = str(e)
                    
                    # 检测 450 违规内容错误
                    if "450" in error_str or "违规" in error_str:
                        print(f"[JinaBrowse] {current_engine} returned 450 content policy error. Switching to fallback engine...", flush=True)
                        last_error = error_str
                        break 
                    
                    # 其他错误正常重试
                    print(f"[JinaBrowse] {current_engine} LLM attempt {attempt + 1}/{max_retry} failed: {e}", flush=True)
                    if attempt == max_retry - 1:
                        last_error = error_str
                        break  
                    time.sleep(random.uniform(1, 4))
            
        print(f"[JinaBrowse] All LLM engines failed. Last error: {last_error}", flush=True)
        return None

    def _extract_info_from_content(self, source_text: str, browse_query: str) -> str:
        """
        Extract relevant information from webpage content using LLM.
        
        Args:
            source_text (str): The webpage content
            browse_query (str): The query to answer
            
        Returns:
            str: The extracted information
        """
        if not tiktoken:
            # If tiktoken is not available, just use the first part
            if len(source_text) > 100000:
                source_text = source_text[:100000] + "...(truncated)"
            
            prompt = f"Please read the source content and answer the following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
            
            output = self._get_llm_response(prompt, temperature=0.0, max_retry=100)
            return output or "Failed to extract information."

        # Use tiktoken to handle long content
        encoding = tiktoken.get_encoding("cl100k_base")
        tokenized_source = encoding.encode(source_text)
        
        if len(tokenized_source) > self.max_tokens_per_chunk:
            # Split into chunks
            num_split = max(2, len(tokenized_source) // self.max_tokens_per_chunk + 1)
            chunk_len = len(tokenized_source) // num_split
            print(f"[JinaBrowse] Content too long ({len(tokenized_source)} tokens), splitting into {num_split} parts", flush=True)
            
            output = "Since the content is too long, the result is split and answered separately. Please combine the results to get the complete answer.\n\n"
            outputs = []
            
            for i in range(num_split):
                start_idx = i * chunk_len
                end_idx = min(start_idx + chunk_len + 1024, len(tokenized_source))
                chunk_text = encoding.decode(tokenized_source[start_idx:end_idx])
                
                prompt = f"Please read the source content and answer the following question:\n--- begin of source content ---\n{chunk_text}\n--- end of source content ---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
                
                chunk_output = self._get_llm_response(prompt, temperature=0.0, max_retry=100)
                outputs.append(chunk_output or "")
            
            for i, chunk_output in enumerate(outputs):
                output += f"--- begin of result part {i+1} ---\n{chunk_output}\n--- end of result part {i+1} ---\n\n"
            
            return output
        else:
            # Content is short enough, process in one go
            prompt = f"Please read the source content and answer the following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
            
            output = self._get_llm_response(prompt, temperature=0.0, max_retry=100)
            return output or "Failed to extract information."

    def forward(self, url: str, query: str = "") -> ToolOutput:
        """
        Browse a webpage and extract relevant information.

        Args:
            url (str): The URL to browse
            query (str): The query to answer based on the webpage content

        Returns:
            ToolOutput: An object containing either the extracted information or an error message.
        """
        try:
            # Validate inputs
            if not url or not isinstance(url, str):
                return ToolOutput(name=self.name or "browse", error="URL is missing, empty, or not a string")
            
            # Default query if empty
            if not query or query.strip() == "":
                query = "Detailed summary of the page."
            
            # Add random delay to avoid rate limiting
            time.sleep(random.uniform(0, 2))
            
            # Read page content
            print(f"[JinaBrowse] Reading page: {url}", flush=True)
            source_text = self._read_page_with_jina(url)
            
            if not source_text or source_text.strip() == "":
                return ToolOutput(name=self.name or "browse", error="Failed to read page content (empty response)")
            
            print(f"[JinaBrowse] Content length: {len(source_text)} characters", flush=True)
            
            # Extract information using LLM
            print(f"[JinaBrowse] Extracting information for query: {query}", flush=True)
            result = self._extract_info_from_content(source_text, query)
            
            if not result or result.strip() == "":
                return ToolOutput(name=self.name or "browse", error="Failed to extract information (empty output)")
            
            print(f"[JinaBrowse] Extraction complete. Result length: {len(result)} characters", flush=True)
            return ToolOutput(name=self.name or "browse", output=result.strip())
            
        except Exception as e:
            error_msg = f"Browse failed for URL '{url}': {str(e)}"
            print(f"[JinaBrowse] Error: {error_msg}", flush=True)
            return ToolOutput(name=self.name or "browse", error=error_msg)

    def __del__(self):
        """Clean up the HTTP client on deletion."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Test the tool
    browse = JinaBrowseTool( generate_engine ="gemini")
    print("Testing Jina Browse Tool...")
    print("=" * 80)
    
    result = browse(url="https://www.baidu.com", query="What is this website about?")
    print(result)
    print("=" * 80)

