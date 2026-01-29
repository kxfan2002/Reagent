import json
import os
import http.client
import time
import random
from typing import List, Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
import requests
import re

SERPER_API_KEY = os.environ.get('SERPER_KEY_ID', '')
SEARCH_API_KEY=os.environ.get('SEARCH_API_KEY', '')
def get_searches_results(queries: List[str], topk: int = 10, engine: str = "serper", max_retry: int = 50) -> str:
    """Get search results for multiple queries using specified search engine."""
    results = []
    for i, query in enumerate(queries):
        result = get_search_results(query, topk=topk, engine=engine, max_retry=max_retry)
        formatted_result = f"--- search result for [{query}] ---\n{result}\n--- end of search result ---"
        results.append(formatted_result)
    return "\n\n".join(results)


def get_search_results(query: str, topk: int = 10, engine: str = "serper", max_retry: int = 50) -> str:
    """Get search results for a single query using specified search engine."""
    if engine == "serper":
        return google_search_with_serp(query, topk=topk, max_retry=max_retry)
    else:
        raise ValueError(f"Unsupported search engine: {engine}")


def contains_chinese_basic(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)

def clean_html_b(text):
    """去掉 <b> 和 </b> 标签"""
    return re.sub(r'</?b>', '', text)

def google_search_with_serp(query: str, topk: int = 10, max_retry: int = 50) -> str:
    """Perform Google search using Serper API."""
    # if not SERPER_API_KEY:
    query = query.replace('"', '').replace("'", '')
    if not SEARCH_API_KEY:
        raise ValueError("SEARCH_API_KEY environment variable is not set")
    
    url = "https://google.serper.dev/search"
    headers = {
        # "X-API-KEY": SEARCH_API_KEY,
        "Authorization": f"Bearer {SEARCH_API_KEY}",
        "Content-Type": "application/json"
    }
    # payload = {
    #     "q": query,
    #     "num": topk
    # }
    payload= {
        "query": query,
        "top_k": topk,
        "api": "bing-search"
    }
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)

            # 检查429 Too Many Requests
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_time = float(retry_after) if retry_after else (3 + random.uniform(1, 4))
                print(f"429 Too Many Requests. Retrying in {wait_time} seconds...", flush=True)
                time.sleep(wait_time)
                continue  # 进入下一次 retry 循环
            
            response.raise_for_status()
            results = response.json()
            if "results" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = []
            
            for page in results["results"][:topk]:
                # 构建snippet内容
                snippet = ""
                if "snippet" in page:
                    snippet = page["snippet"]
                
                # 添加日期信息到snippet中（如果有的话）
                # if "date" in page:
                    # snippet = f"Date published: {page['date']}\n{snippet}"
                
                # 添加来源信息到snippet中（如果有的话）
                # if "source" in page:
                    # snippet = f"Source: {page['source']}\n{snippet}"
                
                # 清理内容
                title = clean_html_b(page.get("title", ""))
                url = page.get("link", "")
                snippet = clean_html_b(page.get("snippet", ""))
                snippet = snippet.replace("Your browser can't play this video.", "")
                
                redacted_version = f"[{title}]({url}) {snippet}"
                web_snippets.append(redacted_version)

            content = "\n\n".join(web_snippets)
            return content
            
        except Exception as e:
            print(f"google_search_with_serp {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return f"No results found for '{query}'. Try with a more general query. Error: {str(e)}"
            time.sleep(3+random.uniform(1, 4))
    
    return f"Search failed after {max_retry} retries for query: '{query}'"


@register_tool("search", allow_overwrite=True)
class WebExplorerSearch(BaseTool):
    name = "search"
    description = "Web search tool that performs batched web searches: supply an array 'queries'; the tool retrieves search results for each query."
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. The queries will be sent to search engine. You will get the brief search results with (title, url, snippet)s for each query."
            },
            "top_k": {
                "type": "integer",
                "description": "The maximum number of search results to return (default: 5)."
            }
        },
        "required": ["queries"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.search_engine = cfg.get("search_engine", "serper") if cfg else "serper"
        self.topk = cfg.get("topk", 10) if cfg else 10
        self.max_retry = cfg.get("max_retry", 50) if cfg else 50

    def call(self, params: Union[str, dict], **kwargs) -> str:
        if "queries" in params:
            queries = params["queries"]
        elif "query" in params:
            queries = params["query"]
        else:
            return "[Search] Invalid request format: Must contain 'queries' field"

        if isinstance(queries, str):
            # Single query (backward compatibility)
            queries = [queries]
        
        if not isinstance(queries, list):
            return "[Search] Error: 'queries' must be a list of strings"
        top_k = self.topk  # 默认

        if "top_k" in params:
            try:
                _topk = int(params["top_k"])
                if _topk > 0:
                    top_k = _topk  # 合法 → 使用用户自定义
                # 不合法 → 忽略，保持 top_k = self.topk
            except:
                pass  # 解析失败 → 忽略，使用默认值
        try:
            result = get_searches_results(
                queries=queries,
                topk=top_k,
                engine=self.search_engine,
                max_retry=self.max_retry
            )
            return result
        except Exception as e:
            return f"[Search] Error: {str(e)}"

if __name__ == "__main__":
    result = WebExplorerSearch().call({"queries": ["What is the capital of July?", "What is the capital of China?"]})
    print(result)