"""
Unified React Agent for multi-dataset evaluation.
Dynamically loads tools based on dataset configuration.
"""
import json
import json5
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import tiktoken
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
from qwen_agent.utils.utils import format_as_text_message, merge_generate_cfgs
from prompt_generator import generate_system_prompt, get_dataset_config
import time
import asyncio

# Import all possible tools
from tool_python_executor import PythonExecutor
from tool_search import WebExplorerSearch
from tool_browse import WebExplorerBrowse
from tool_audio2text import Audio2Text
from tool_image2text import Image2Text
from tool_filereader import FileReader
from auto_judge import compute_score_genrm, simple_em_score

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

TRUNCATED_MESSAGE = """
--- Maximum Length Limit Reached ---
You have reached the maximum length limit. 
The response is truncated."""

FINAL_MESSAGE = """
--- Final Step Reached ---
Now you reach the final step. 
You are forbidden to call any tools.
You must offer your final answer now."""

# All available tool instances
ALL_TOOLS = {
    "python": PythonExecutor(),
    "search": WebExplorerSearch(),
    "browse": WebExplorerBrowse(),
    "image2text": Image2Text(),
    "audio2text": Audio2Text(),
    "file_reader": FileReader()
}

import random


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 dataset_name: str = "gaia",
                 **kwargs):
        """
        Initialize the Multi-Turn React Agent.
        
        Args:
            function_list: List of function names to use (if None, loaded from dataset config)
            llm: LLM configuration
            dataset_name: Name of the dataset (used to load configuration)
            **kwargs: Additional arguments
        """
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.dataset_name = dataset_name
        
        # Load dataset configuration
        self.dataset_config = get_dataset_config(dataset_name)
        
        # Use provided function_list or load from config
        if function_list is None:
            function_list = self.dataset_config['tools']
        
        self.function_list = function_list
        
        # Build tool map with only the required tools
        self.tool_map = {name: ALL_TOOLS[name] for name in function_list if name in ALL_TOOLS}
        
        # Generate system prompt based on tools
        self.system_prompt = generate_system_prompt(function_list)
        
        # Get max LLM calls from config
        self.max_llm_call_per_run = self.dataset_config.get('max_llm_calls', 100)
        
        print(f"Initialized agent for dataset: {dataset_name}")
        print(f"Tools: {function_list}")
        print(f"Max LLM calls: {self.max_llm_call_per_run}")
    
    def sanity_check_output(self, content):
        """Check if the output contains thinking tags."""
        return "<think>" in content and "</think>" in content
    
    def call_server(self, msgs, planning_port, max_tries=10, token_num=8192):
        """
        Call the vLLM server to generate a response.
        
        Args:
            msgs: Messages to send to the server
            planning_port: Port number of the vLLM server
            max_tries: Maximum number of retry attempts
            token_num: Number of tokens already used
            
        Returns:
            Generated content string
        """
        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1 
        
        max_token = min(32760-token_num, 8192)
        for attempt in range(max_tries):
            try:
                print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=max_token
                )
                content = chat_response.choices[0].message.content
                if content and content.strip():
                    print("--- Service call successful, received a valid response ---")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response.")

            except (APIError, APIConnectionError, APITimeoutError) as e:
                status_code = getattr(e, "status_code", None)
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
                # Handle 429 rate limit with exponential backoff
                if status_code == 429:
                    sleep_time = min(base_sleep_time * (2 ** (attempt + 2)) + random.uniform(1, 5) + 5, 120)
                    print(f"Rate limit hit (429). Cooling off for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
                    continue
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
        
        return "vllm server error!!!"

    def add_auto_judge(self, result, auto_judge, judge_engine, messages, question, answer):
        """Add automatic judgment result to the return dictionary."""
        if auto_judge and answer:
            try:
                # Use prediction instead of entire conversation history
                prediction = result.get("prediction", "")
                if not prediction:
                    print("Warning: No prediction found for auto judge")
                    result["auto_judge"] = {"error": "No prediction found", "score": 0}
                    return result
                
                judge_result = compute_score_genrm(
                    prediction=prediction,
                    ground_truth=answer,
                    question=question,
                    engine=judge_engine
                )
                result["auto_judge"] = judge_result
                print(f"Auto Judge Score: {judge_result['score']}, Prediction: '{prediction[:100]}...', Ground Truth: '{answer}'")
            except Exception as e:
                print(f"Auto judge failed: {e}")
                result["auto_judge"] = {"error": str(e), "score": 0}
        return result

    def count_tokens(self, messages, model="gpt-4o"):
        """Count the number of tokens in messages."""
        try: 
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 
        except Exception as e: 
            tokenizer = tiktoken.encoding_for_model(model)
        
        full_message = [Message(**x) for x in messages]
        full_prompt = build_text_completion_prompt(
            full_message, 
            allow_special=True,
            default_system=self.system_prompt
        )
        
        return len(tokenizer.encode(full_prompt))

    def _run(self, data: str, model: str, auto_judge: bool = False, judge_engine: str = "deepseekchat", **kwargs) -> List[List[Message]]:
        """
        Run the agent on a single data item.
        
        Args:
            data: Data item dictionary containing question and metadata
            model: Model path or name
            auto_judge: Whether to enable automatic judgment
            judge_engine: Engine to use for automatic judgment
            **kwargs: Additional arguments
            
        Returns:
            Result dictionary with prediction and metadata
        """
        self.model = model
        
        # Try to get question from different possible field names
        question = data['item'].get('Question', '') or data['item'].get('question', '')
        if not question:
            # If no question field, try to extract from messages
            if 'messages' in data['item']:
                try:
                    raw_msg = data['item']['messages'][1]["content"] 
                    question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
                except Exception as e:
                    print(f"Failed to extract question from messages: {e}")
                    question = ""
        
        if not question:
            raise ValueError("No question found in data item")

        start_time = time.time()
        planning_port = data['planning_port']
        
        # Try to get answer from different possible field names
        answer = data['item'].get('Final answer', '') or data['item'].get('answer', '')
        
        # Note: File attachments are handled in run_multi_react_py.py (only for rollout_idx == 1)
        # to avoid duplicate processing across multiple rollouts
        
        self.user_prompt = question
        messages = [
            {"role": "system", "content": self.system_prompt}, 
            {"role": "user", "content": question}
        ]
        
        num_llm_calls_available = self.max_llm_call_per_run
        round = 0
        timeout_seconds = self.dataset_config.get('timeout_seconds', 9000)  # 150 minutes default
        
        while num_llm_calls_available > 0:
            # Check whether timeout is reached
            if time.time() - start_time > timeout_seconds:
                prediction = f'No answer found after {timeout_seconds//60} minutes'
                termination = f'No answer found after {timeout_seconds//60} minutes'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                result = self.add_auto_judge(result, auto_judge, judge_engine, messages, question, answer)
                return result
            
            round += 1
            num_llm_calls_available -= 1
            content = self.call_server(messages, planning_port)
            print(f'Round {round}: {content}')
            
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            messages.append({"role": "assistant", "content": content.strip()})
            
            # Check if there is a tool call
            has_tool_call = False
            if '<tool_call>' in content and '</tool_call>' in content:
                has_tool_call = True
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})
                    result = self.custom_call_tool(tool_name, tool_args)
                except:
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                    
                result = "<tool_response>\n" + result + "\n</tool_response>"
                print(result)
                messages.append({"role": "user", "content": result})
            
            # If no tool call, stop
            if not has_tool_call:
                termination = 'no_tool_call'
                break
                
            if num_llm_calls_available <= 0:
                # If calls exhausted, add final message and require answer
                messages.append({"role": "user", "content": FINAL_MESSAGE})
                content = self.call_server(messages, planning_port)
                messages.append({"role": "assistant", "content": content.strip()})
                prediction = messages[-1]['content']
                termination = 'exceed_llm_calls'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                result = self.add_auto_judge(result, auto_judge, judge_engine, messages, question, answer)
                return result

            max_tokens = self.dataset_config.get('max_token_limit', 24477)
            token_count = self.count_tokens(messages)
            print(f"round: {round}, token count: {token_count}")

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")
                
                messages[-1]['content'] = TRUNCATED_MESSAGE + FINAL_MESSAGE
                content = self.call_server(messages, planning_port, token_count)
                messages.append({"role": "assistant", "content": content.strip()})
                prediction = messages[-1]['content']
                termination = 'token_limit_reached'
                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination
                }
                result = self.add_auto_judge(result, auto_judge, judge_engine, messages, question, answer)
                return result

        # Handle final result
        prediction = messages[-1]['content']
        if termination != 'no_tool_call':
            if num_llm_calls_available <= 0:
                termination = 'exceed_llm_calls'
            else:
                termination = 'unknown'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination
        }
        result = self.add_auto_judge(result, auto_judge, judge_engine, messages, question, answer)
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        """Call a tool by name with provided arguments."""
        if tool_name in self.tool_map:
            tool_args["params"] = tool_args
            raw_result = self.tool_map[tool_name].call(tool_args, **kwargs)
            result = raw_result
            return result
        else:
            return f"Error: Tool {tool_name} not found. Available tools: {list(self.tool_map.keys())}"

