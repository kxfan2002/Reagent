import os
import time
import random
import requests
from typing import Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI
import tiktoken


JINA_API_KEY = os.getenv("JINA_API_KEYS", "")

class LLMError(Exception):
    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        self.message = message
        super().__init__(f"LLMError {status_code}: {message}")

def call_llm_once(engine: str, query: str):
    if engine == "geminiflash":
        return get_geminiflash_response(query, temperature=0.0)
    elif engine == "deepseekchat":
        return get_deepseekchat_response(query, temperature=0.0)
    elif engine == "openai":
        return get_openai_response(query, temperature=0.0)
    else:
        raise ValueError(f"Unsupported generate engine: {engine}")


def call_llm_with_retry(query: str, engines: list[str], max_retry: int = 5):
    last_error_msg = "Unknown Error"
    
    for engine in engines:
        retry_cnt = 0
        while retry_cnt < max_retry:
            try:
                # 核心调用
                return call_llm_once(engine, query)

            except LLMError as e:
                last_error_msg = f"{engine}: {e.message}"
                
                # 1. 450 直接失败，停止所有 engine
                if e.status_code == 450:
                    return f"Configuration Error: {e.message}. Please check API keys."

                # 2. 500 直接换 engine (跳出 while，进入下一个 for 循环)
                if e.status_code == 500:
                    print(f"[WARN] {engine} hit 500, switching engine...", flush=True)
                    break 

                # 3. 429 或 返回为空(460)：当前 engine 重试
                if e.status_code in (429, 460):
                    retry_cnt += 1
                    print(f"[WARN] {engine} error {e.status_code}, retrying {retry_cnt}/{max_retry}", flush=True)
                    time.sleep(random.uniform(2, 5))
                    continue
                
                # 其他 LLMError，默认也尝试在当前 engine 重试
                retry_cnt += 1
                time.sleep(1)

            except Exception as e:
                # 捕获非 LLMError 的系统异常（如网络超时等）
                retry_cnt += 1
                last_error_msg = str(e)
                time.sleep(2)

    # 4. 最终工具不 raise，而是返回报错字符串给 Agent
    return f"All LLM engines failed to process the request. Last error: {last_error_msg}"

def call_llm_once(engine: str, query: str):
    # 注意：确保子函数内部不要再写 for retry 循环，只负责单次调用并抛出 LLMError
    if engine == "geminiflash":
        return get_geminiflash_response(query, max_retry=1) 
    elif engine == "deepseekchat":
        return get_deepseekchat_response(query, max_retry=1)
    elif engine == "openai":
        return get_openai_response(query, max_retry=1)
    else:
        return "Unsupported engine specified."


def get_geminiflash_response(query: str, temperature: float = 0.0, max_retry: int = 5) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    
    if not api_key:
        raise LLMError(450, "GEMINI_API_KEY not set")

    client = OpenAI(api_key=api_key, base_url=api_base)

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=8192
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError(460, "empty response")
        return content

    except Exception as e:
        msg = str(e).lower()

        # 限流 / 配额 → 换模型
        if "429" in msg or "rate" in msg or "quota" in msg:
            raise LLMError(429, msg)

        # 服务端错误 → 直接失败
        if "500" in msg or "internal" in msg:
            raise LLMError(500, msg)




def get_deepseekchat_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from DeepSeek Chat model using standard OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    if not api_key:
        raise LLMError(450, "DEEPSEEK_API_KEY not set")
    
    client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
    for retry_cnt in range(max_retry):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": query}],
                temperature=temperature,
                max_tokens=8192
                # max_tokens=32768
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMError(460, "empty response")
            return content
        except Exception as e:
            msg = str(e).lower()

            # 限流 / 配额 → 换模型
            if "429" in msg or "rate" in msg or "quota" in msg:
                raise LLMError(429, msg)

            # 服务端错误 → 直接失败
            if "500" in msg or "internal" in msg:
                raise LLMError(500, msg)

            # 网络抖动 / timeout / JSON 等 → 可重试
            if retry_cnt == max_retry - 1:
                return None

            time.sleep(random.uniform(2, 4))



def get_openai_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from OpenAI API."""
    api_key = os.environ.get("API_KEY")
    url_llm = os.environ.get("API_BASE")
    model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
    
    if not api_key or not url_llm:
        raise LLMError(450, "OpenAI config not set")
        
    client = OpenAI(
        api_key=api_key,
        base_url=url_llm,
    )
    
    for attempt in range(max_retry):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                temperature=temperature
            )
            content = chat_response.choices[0].message.content
            if not content:
                raise LLMError(460, "empty response from gpt")
            return content
        except Exception as e:
            msg = str(e).lower()

            # 限流 / 配额 → 换模型
            if "429" in msg or "rate" in msg or "quota" in msg:
                raise LLMError(429, msg)

            # 服务端错误 → 直接失败
            if "500" in msg or "internal" in msg:
                raise LLMError(500, msg)

            # 网络抖动 / timeout / JSON 等 → 可重试
            if attempt == max_retry - 1:
                return None

            time.sleep(random.uniform(2, 4))


def jina_readpage(url: str, max_retry: int = 10) -> str:
    """Read webpage content using Jina service."""
    if not JINA_API_KEY:
        return "[browse] JINA_API_KEYS environment variable is not set."
    headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
    proxy="http://10.70.16.106:3128"
    proxies = {"http": proxy, "https": proxy}
    for attempt in range(max_retry):
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=50,
                proxies=proxies
            )
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            elif response.status_code == 429:
                wait_time = 4+random.uniform(2,4)
                print(f"JINA 429 Too Many Requests. Retrying in {wait_time} seconds...", flush=True)
                time.sleep(wait_time)
                continue  # 进入下一次 retry 循环
            elif response.status_code == 422:
                # 422 表示无法访问目标 URL，重试无意义，直接返回错误信息
                error_msg = response.text if hasattr(response, 'text') else "Failed to access URL"
                print(f"Jina API 422 error: {error_msg}", flush=True)
                return f"[browse] Failed to access URL: {error_msg}"
            else:
                print(f"Jina API error (status {response.status_code}): {response.text}", flush=True)
                # 对于其他错误，如果是最后一次尝试，返回错误信息
                if attempt == max_retry - 1:
                    return f"[browse] Failed to read page. Status code: {response.status_code}"
                time.sleep(random.uniform(2, 4))
                continue
        except requests.exceptions.RequestException as e:
            # 网络请求异常（超时、连接错误等）
            print(f"jina_readpage {attempt} network error: {e}", flush=True)
            if attempt == max_retry - 1:
                return "[browse] Failed to read page due to network error."
            time.sleep(random.uniform(2, 4))
        except Exception as e:
            # 其他未知异常
            print(f"jina_readpage {attempt} error: {e}", flush=True)
            if attempt == max_retry - 1:
                return "[browse] Failed to read page."
            time.sleep(random.uniform(1, 3))
                
    return "[browse] Failed to read page."


def get_browse_results(url: str, browse_query: str, read_engine: str = "jina", generate_engine: str = "deepseekchat", max_retry: int = 5) -> str:
    """Get browse results by reading webpage and extracting relevant information."""
    time.sleep(random.uniform(0, 16))
    print("+++++ in get_browse_results ++++++")
    print("browse query:"+browse_query)
    # Read webpage content
    source_text = ""
    for retry_cnt in range(max_retry):
        try:
            if read_engine == "jina":
                source_text = jina_readpage(url, max_retry=20)
                print("#"*20)
                print(source_text)
                print("#"*20)
            else:
                raise ValueError(f"Unsupported read engine: {read_engine}")
            break
        except Exception as e:
            print(f"Read {read_engine} {retry_cnt} error: {e}, url: {url}", flush=True)
            if any(word in str(e) for word in ["Client Error"]):
                return "Access to this URL is denied. Please try again."
            time.sleep(random.uniform(16, 64))

    if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
        print(f"Browse error with empty source_text.", flush=True)
        return "Browse error. Jina reader fails to read page Please try again."
    
    browse_query = browse_query.replace('"', '').replace("'", '')
    query = f"Please read the source content and answer a following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
    
    # 处理长内容分块（仿照deep_research_utils.py的逻辑）
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_source_text = encoding.encode(source_text)
    
    if len(tokenized_source_text) > 95000:  # 使用与原代码相同的token限制
        output = "Since the content is too long, the result is split and answered separately. Please combine the results to get the complete answer.\n"
        num_split = max(2, len(tokenized_source_text) // 95000 + 1)
        chunk_len = len(tokenized_source_text) // num_split
        print(f"Browse too long with length {len(tokenized_source_text)}, split into {num_split} parts, with each part length {chunk_len}", flush=True)
        
        outputs = []
        for i in range(num_split):
            start_idx = i * chunk_len
            end_idx = min(start_idx + chunk_len + 1024, len(tokenized_source_text))
            source_text_i = encoding.decode(tokenized_source_text[start_idx:end_idx])
            query_i = f"Please read the source content and answer a following question:\n--- begin of source content ---\n{source_text_i}\n--- end of source content ---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
            
            # if generate_engine == "geminiflash":
            #     output_i = get_geminiflash_response(query_i, temperature=0.0, max_retry=1)
            # elif generate_engine == "deepseekchat":
            #     output_i = get_deepseekchat_response(query_i, temperature=0.0, max_retry=1)
            # elif generate_engine == "openai":
            #     output_i = get_openai_response(query_i, temperature=0.0, max_retry=1)
            # else:
            #     raise ValueError(f"Unsupported generate engine: {generate_engine}")
            output_i = call_llm_with_retry(
                query_i,
                engines=["deepseekchat", "geminiflash","openai"],
                max_retry=20
            )

            
            outputs.append(output_i or "")
        
        for i in range(num_split):
            output += f"--- begin of result part {i+1} ---\n{outputs[i]}\n--- end of result part {i+1} ---\n\n"
    else:
        # if generate_engine == "geminiflash":
        #     output = get_geminiflash_response(query, temperature=0.0, max_retry=1)
        # elif generate_engine == "deepseekchat":
        #     output = get_deepseekchat_response(query, temperature=0.0, max_retry=1)
        # elif generate_engine == "openai":
        #     output = get_openai_response(query, temperature=0.0, max_retry=1)
        # else:
        #     raise ValueError(f"Unsupported generate engine: {generate_engine}")
        output = call_llm_with_retry(
            query,
            engines=["openai", "deepseekchat", "geminiflash"],
            max_retry=20
        )

    
    if output is None or output.strip() == "":
        print(f"Browse error with empty output.", flush=True)
        return "Browse error. Summary model fails to give summary. Please try again."
    
    return output


@register_tool("browse", allow_overwrite=True)
class WebExplorerBrowse(BaseTool):
    name = "browse"
    description = "Browse a webpage and extract relevant information based on a specific query."
    parameters = {
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

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.read_engine = cfg.get("read_engine", "jina") if cfg else "jina"
        self.generate_engine = cfg.get("generate_engine", "deepseekchat") if cfg else "deepseekchat"
        self.max_retry = cfg.get("max_retry", 3) if cfg else 3

    def call(self, params: Union[str, dict], **kwargs) -> str:
        print("+++++ in browse call ++++++")
        try:
            url = params["url"]
            # query = params["query"]
        except:
            return "[Browse] Invalid request format: Input must be a JSON object containing 'url' and 'query' fields"
        
        if "query" in params:
            query = params["query"]
        elif "queries" in params:
            query = params["queries"]
        else:
            return "[Browse] Invalid request format: Must contain 'query' field"

        if not url or not isinstance(url, str):
            return "[Browse] Error: 'url' is missing, empty, or not a string"
        
        if not isinstance(query, str):
            return "[Browse] Error: 'query' is missing or not a string"
        
        if query == "":
            query = "Detailed summary of the page."

        try:
            print("+++++ before get_browse_results ++++++")
            result = get_browse_results(
                url=url,
                browse_query=query,
                read_engine=self.read_engine,
                generate_engine=self.generate_engine,
                max_retry=self.max_retry
            )
            print("+++++ after get_browse_results ++++++")
            print(f'Browse Summary Length {len(result)}; Browse Summary Content {result}')
            return result.strip()
            
        except Exception as e:
            return f"[Browse] Error: {str(e)}"

if __name__ == "__main__":
    result = WebExplorerBrowse().call({"url": "https://journals.le.ac.uk/index.php/jist/article/view/733", "query": "What is the volume in cubic meters of the fish bag calculated in this paper? Look for any calculations involving fish bag dimensions, volume, or cubic meters."})
    print(result)
    #{\"url\": \"https://ruffwear.com/blogs/explored\", \"query\": \"What story or blog post was added on December 8th, 2022? Please search through all the blog posts looking for a date of December 8th, 2022.