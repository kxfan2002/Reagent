"""
Web search tool implementation using Bing Search backend.

This module provides a web search tool that integrates with a search API
to perform batch web searches with automatic retry and rate limiting handling.
"""

import os
import re
import time
import random
from typing import Any

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

# Configuration constants
REFERENCE_COUNT = 8  # Default number of search results to return
DEFAULT_SEARCH_ENGINE_TIMEOUT = 30  # Timeout in seconds for search requests
SEARCH_API_ENDPOINT = os.environ.get(
    "SEARCH_API_ENDPOINT",
    "https://aigc.sankuai.com/v1/friday/api/search"  # Default endpoint
)

# Retry configuration
MAX_RETRY_ATTEMPTS = 200  # Maximum total retry attempts (for rate limiting scenarios)
MAX_RETRY_500_ERRORS = 3  # Maximum retry attempts for 500 internal server errors
MIN_RETRY_WAIT_TIME = 1.0  # Minimum wait time in seconds before retry
MAX_RETRY_JITTER = 4.0  # Maximum random jitter added to wait time


def clean_html_tags(text: str) -> str:
    """
    Remove HTML tags from text (e.g., <b> and </b>).
    
    Args:
        text: Input text that may contain HTML tags
        
    Returns:
        str: Text with HTML tags removed
    """
    return re.sub(r'</?b>', '', text)


class WebSearchTool(Tool):
    """A tool for performing web searches using Bing Search backend."""

    NAME = "search"
    DESCRIPTION = "Web search tool that performs web searches: the tool retrieves search results for each query."

    def __init__(
        self, 
        name: str = NAME, 
        description: str = DESCRIPTION, 
        timeout: float = DEFAULT_SEARCH_ENGINE_TIMEOUT, 
        reference_count: int = REFERENCE_COUNT,
        max_retry: int = MAX_RETRY_ATTEMPTS
    ):
        """
        Initialize the WebSearch tool.

        Args:
            name (str): The name of the tool.
            description (str): A description of the tool's purpose.
            timeout (float): Maximum time in seconds to wait for search results.
            reference_count (int): Number of results to return per query.
            max_retry (int): Maximum number of retry attempts for rate limiting scenarios.
                            Note: 500 errors have a separate retry limit (MAX_RETRY_500_ERRORS).
        """
        self.timeout = timeout
        self.reference_count = reference_count
        self.max_retry = max_retry
        self._init_client()
        super().__init__(name=name, description=description)

    def _init_client(self):
        """
        Initialize the HTTP client for making requests.
        """
        self.client = httpx.Client()

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Web search tool that performs batched web searches: supply an array 'queries'; the tool retrieves search results for each query.",
                "parameters": {
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
                    "required": ["queries"]
                }
            }
        }

    def _search_with_api(self, query: str, top_k: int | None = None):
        """
        Search with the configured API and return the contexts.
        
        Args:
            query (str): Search query string
            top_k (int | None): Number of results to return (optional)
            
        Returns:
            list: List of search result dictionaries
        """
        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            raise ValueError("SEARCH_API_KEY environment variable is not set")
        
        cleaned_query = query.replace('"', '').replace("'", '')
        if cleaned_query != query:
            print(f"[WebSearch] Query cleaned: '{query}' -> '{cleaned_query}'", flush=True)

        actual_top_k = top_k if top_k is not None else self.reference_count
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": cleaned_query,
            "top_k": actual_top_k,
            "api": "bing-search"
        }
        
        retry_500_count = 0  # Counter for 500 internal server errors

        for retry_cnt in range(self.max_retry):
            try:
                response = self.client.post(
                    url=SEARCH_API_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Handle 429 Too Many Requests error with exponential backoff
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = MIN_RETRY_WAIT_TIME
                    else:
                        wait_time = MIN_RETRY_WAIT_TIME
                    
                    # Ensure minimum wait time
                    wait_time = max(wait_time, MIN_RETRY_WAIT_TIME)
                    # Add random jitter to prevent thundering herd
                    wait_time += random.uniform(MIN_RETRY_WAIT_TIME, MAX_RETRY_JITTER)
                    
                    print(f"[WebSearch] 429 Too Many Requests for query '{cleaned_query}'. Waiting {wait_time:.1f}s before retry ({retry_cnt + 1}/{self.max_retry})...", flush=True)
                    time.sleep(wait_time)
                    continue  # Retry instead of raising exception
                
                # Handle 500 Internal Server Error with limited retries
                if response.status_code == 500:
                    retry_500_count += 1
                    print(f"[WebSearch] 500 Internal Server Error for query '{cleaned_query}'. Retry {retry_500_count}/{MAX_RETRY_500_ERRORS}...", flush=True)
                    
                    if retry_500_count >= MAX_RETRY_500_ERRORS:
                        # Max 500 error retries reached, raise exception
                        print(f"[WebSearch] 500 error persisted after {MAX_RETRY_500_ERRORS} retries for query '{cleaned_query}'", flush=True)
                        raise Exception(f"500 Internal Server Error after {MAX_RETRY_500_ERRORS} retries for query: '{cleaned_query}'")
                    
                    # Wait before retry with jitter
                    time.sleep(MIN_RETRY_WAIT_TIME + random.uniform(0, 2))
                    continue
                
                response.raise_for_status()
                
                json_content = response.json()
                
                if "results" not in json_content:
                    raise Exception(f"No results found for query: '{query}'. Use a less specific query.")
                
                contexts = json_content["results"][:actual_top_k]
                return contexts
                
            except Exception as e:
                error_msg = str(e)
                # If 500 error limit reached, re-raise immediately without further retry
                if "500 Internal Server Error after" in error_msg and "retries for query" in error_msg:
                    raise
                # 429 errors are handled above, this handles other errors
                if isinstance(e, Exception) and "429" not in error_msg:
                    print(f"[WebSearch] Retry {retry_cnt + 1}/{self.max_retry} failed: {e}", flush=True)
                    if retry_cnt == self.max_retry - 1:
                        # Last retry failed
                        raise
                    # Wait before retry with jitter (minimum 3 seconds for network errors)
                    time.sleep(3 + random.uniform(MIN_RETRY_WAIT_TIME, MAX_RETRY_JITTER))
        
        raise Exception(f"Search failed after {self.max_retry} retries for query: '{query}'")

    def forward(self, queries: list[str] | str, top_k: int | None = None) -> ToolOutput:
        """
        Execute searches with the given queries.

        Args:
            queries (list[str] | str): Query or list of queries to be submitted to the search engine.
                                       If a single string is provided, it will be converted to a list for backward compatibility.
            top_k (int | None): Number of results to return per query (optional)

        Returns:
            ToolOutput: An object containing either the search results or an error message.
        """
        try:
            assert self.client is not None, "Search Client not initialized"
            
            # Handle backward compatibility: convert single string to list
            if isinstance(queries, str):
                queries = [queries]
            
            if not isinstance(queries, list):
                return ToolOutput(
                    name=self.name or "search",
                    error="'queries' must be a list of strings or a single string"
                )
            
            
            all_results = []
            
            # Process each query
            for query in queries:
                if not isinstance(query, str):
                    print(f"[WebSearch] Skipping non-string query: {query}", flush=True)
                    continue
                
                try:
                    contexts = self._search_with_api(query, top_k)
                    
                    # Format results for this query
                    web_snippets = []
                    for page in contexts:
                        # Clean HTML tags from title and snippet
                        title = clean_html_tags(page.get("title", ""))
                        snippet = clean_html_tags(page.get("snippet", ""))
                        link = page.get("link", "")
                        
                        # Remove unwanted content
                        snippet = snippet.replace("Your browser can't play this video.", "")
                        
                        # Format as markdown-style link with snippet
                        formatted_result = f"[{title}]({link}) {snippet}"
                        
                        web_snippets.append(formatted_result)
                    
                    # Join results for this query
                    query_results = "\n\n".join(web_snippets)
                    
                    # Add query header
                    formatted_query_result = f"--- search result for [{query}] ---\n{query_results}\n--- end of search result ---"
                    all_results.append(formatted_query_result)
                    
                except Exception as e:
                    error_msg = f"--- search result for [{query}] ---\nSearch failed: {str(e)}\n--- end of search result ---"
                    all_results.append(error_msg)
                    print(f"[WebSearch] Query '{query}' failed: {e}", flush=True)
            
            if not all_results:
                return ToolOutput(
                    name=self.name or "search",
                    error=f"No results found for queries: {queries}"
                )
            
            # Join all query results with double newlines
            result_text = "\n\n".join(all_results)
            
            return ToolOutput(name=self.name or "search", output=result_text)
            
        except Exception as e:
            error_msg = f"Search failed for queries {queries}. Error: {str(e)}"
            return ToolOutput(name=self.name or "search", error=error_msg)

    def __del__(self):
        """Clean up the HTTP client on deletion."""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Test the tool
    search = WebSearchTool()
    print("Testing Web Search Tool...")
    print("=" * 80)
    
    # Test 1: Single query (backward compatibility)
    print("\nTest 1: Single query string (backward compatibility)")
    result = search(queries='"under 50 people" "few households" village Iran 2006 census')
    print(result)
    print("=" * 80)
    
    # Test 2: Multiple queries (new feature)
    # print("\nTest 2: Multiple queries array")
    # result = search(queries=["What is Python programming?", "Who created Python?"])
    # print(result)
    # print("=" * 80)

