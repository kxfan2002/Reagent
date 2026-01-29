"""
Tool-based environment for agent training with reward model integration.

This module provides an environment where agents can interact with various tools
and receive rewards based on their performance and reasoning quality.
"""

import json
import queue
import time
import warnings
from typing import Any
import requests
import re

from rllm.environments.base.base_env import BaseEnv
from rllm.rewards.reward_fn import RewardFunction, zero_reward
from rllm.tools.multi_tool import MultiTool
from rllm.tools.tool_base import Tool

# Configuration constants for reward model
DEFAULT_REWARD_MODEL_LAMBDA = 0.3  # Default weight for reward model score
DEFAULT_MAX_TOKENS = 1200  # Default max tokens for reward model response
MAX_TOKENS_WITH_CRITIQUE = 1400  # Max tokens when requesting critique
REWARD_MODEL_TIMEOUT = 60  # Timeout in seconds for reward model API calls
REWARD_MODEL_RETRY_DELAY = 2  # Delay in seconds between retry attempts
TOOL_EXECUTION_TIMEOUT = 600  # Timeout in seconds for tool execution threads

# Score validation bounds
MIN_REWARD_SCORE = 0.0
MAX_REWARD_SCORE = 1.0

REWARD_MODEL_PROMPT = """
You are an expert agent tool use evaluator. You must strictly follow the output format below:

<think>
Provide a comprehensive analysis of the entire reasoning trajectory. Focus specifically on the agent's reasoning quality and its tool-usage behavior across ANY type of tool.

Key points to evaluate (for all tasks and all tools):
- Whether the agent correctly decided when to call tools. Over-reliance on tools for trivial reasoning is bad; failing to call tools when necessary is also bad.
- Whether the agent misused tools (e.g., calling an irrelevant tool, giving incorrectly formatted arguments, hallucinating tool inputs or filenames, making repeated tool calls without new purpose).
- Whether the agent understood tool limitations (e.g., tool outputs may be incomplete, noisy, or partial; tools cannot access nonexistent resources).
- Whether the agent improved its reasoning over time (e.g., corrected wrong assumptions, avoided repeated mistakes, verified hypotheses when possible).
- Whether the agent avoided unverified guesses. Hypotheses without verification are harmful.
- Whether the agent avoided fabricating tool results, file names, object identifiers, or other non-existent content.

If uncertain, identify potential harmful reasoning patterns: unnecessary tool calls, missing essential tool calls, uncritical acceptance of tool output, faulty logical jumps, or incorrect assumptions about tool capabilities.
Never mention the true answer. Only evaluate the reasoning process and tool use.
</think>

<critique>
Provide a succinct, specific, and actionable summary of issues in the agent's reasoning and tool use. This section will be shown to the agent, so it must be concise and clearly highlight:

- Incorrect, unnecessary, missing, or repeated tool calls.
- Incorrect assumptions, unverified reasoning, or blind trust in tool results.
- Any improper handling of tool limitations or constraints.
- Any hallucinated tool arguments, filenames, or resource identifiers.
- Unlogical reasoning.
Do NOT provide the correct answer or hints toward it.
</critique>

<score>
A single float between 0 and 1 representing the overall quality of the reasoning and tool use.
0 means completely incorrect or harmful reasoning; 1 means flawless reasoning with appropriate, precise, and well-justified tool use.
</score>

Strict requirements:
- The output must contain **exactly three blocks**: <think>, <critique>, <score>.
- The evaluation must focus solely on reasoning and tool-use quality.
- Never reveal the correct answer.
"""

class ToolEnvironment(BaseEnv):
    """
    A simple environment for tool-based agents that provides questions and evaluates responses.
    """

    def __init__(self, task: dict | None = None, tools: list[str] | None = None, 
                 tool_map: dict[str, type[Tool]] | None = None, 
                 reward_fn: RewardFunction | None = None, 
                 max_steps=10,
                 reward_model_config: dict | None = None,
                 ):
        """
        Initialize the ToolEnvironment.

        Args:
            task: Task information for the environment.
            tools: List of tool names to look up in the registry (legacy behavior).
            tool_map: Dictionary mapping tool names to Tool classes (new behavior).
            reward_fn: Reward function to use for evaluation.
            max_steps: Maximum number of steps allowed in the environment.
            reward_model_config: Configuration for optional external reward model integration.
        """
        if tool_map is not None and tools is not None:
            raise ValueError("Cannot specify both 'tools' and 'tool_map' parameters")

        self.step_count = 0
        self.max_steps = max_steps

        # Initialize MultiTool with either tools or tool_map
        if tool_map is not None:
            self.tools = MultiTool(tool_map=tool_map)
        elif tools is not None:
            self.tools = MultiTool(tools=tools)
        else:
            self.tools = MultiTool(tools=[])

        self.task = task
        if reward_fn is None:
            warnings.warn("No reward function specified, will get 0 reward.", stacklevel=2)
            self.reward_fn = zero_reward
        else:
            self.reward_fn = reward_fn
            
        # Initialize reward model configuration
        self.reward_model_config = reward_model_config or {}
        self.enable_reward_model = self.reward_model_config.get("enable", False)
        self.reward_model_url = self.reward_model_config.get("url", None)
        self.reward_model_lambda = self.reward_model_config.get("lambda_weight", DEFAULT_REWARD_MODEL_LAMBDA)

    def _call_reward_model(self, question: str, answer: str, max_retries: int = 5) -> float:
        """
        Call the reward model to get a quality score.
        
        Args:
            question: The question text
            answer: The answer text to evaluate
            max_retries: Maximum number of retry attempts
            
        Returns:
            float: Score between 0 and 1
        """
        if not self.reward_model_url:
            warnings.warn("Reward model URL not configured, returning 0.0")
            return MIN_REWARD_SCORE
        
        # Build complete chat completions URL
        chat_url = f"{self.reward_model_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # Build prompt using the REWARD_MODEL_PROMPT template
        system_prompt = REWARD_MODEL_PROMPT
        
        prompt = f"Question: {question}\n\nAgent Trajectory: {answer}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        
        # Add model name to payload if specified in config
        reward_model_name = self.reward_model_config.get("model")
        if reward_model_name:
            payload["model"] = reward_model_name
        
        for attempt in range(max_retries):
            try:
                print(f"Calling reward model (attempt {attempt + 1}/{max_retries})...", flush=True)
                http_response = requests.post(chat_url, headers=headers, json=payload, timeout=REWARD_MODEL_TIMEOUT)
                http_response.raise_for_status()
                
                result = http_response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    print(f"Attempt {attempt + 1}: Empty response from reward model", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(REWARD_MODEL_RETRY_DELAY)
                        continue
                    return MIN_REWARD_SCORE
                
                # Extract score from <score> tag
                score_match = re.search(r'<score>\s*([\d.]+)\s*</score>', content, re.DOTALL | re.IGNORECASE)
                
                if score_match:
                    score_str = score_match.group(1).strip()
                    score = float(score_str)
                    
                    # Validate score range
                    if MIN_REWARD_SCORE <= score <= MAX_REWARD_SCORE:
                        print(f"Successfully got reward model score: {score}", flush=True)
                        return score
                    else:
                        warnings.warn(f"Reward model score {score} out of range [{MIN_REWARD_SCORE},{MAX_REWARD_SCORE}], retrying...")
                else:
                    print(f"Attempt {attempt + 1}: No <score> tag found in reward model response", flush=True)
                    print(f"Reward model response: {content}...", flush=True)
                    if attempt < max_retries - 1:
                        print("Retrying reward model call...", flush=True)
                        time.sleep(REWARD_MODEL_RETRY_DELAY)
                        continue
                        
            except requests.exceptions.RequestException as e:
                warnings.warn(f"Error calling reward model (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(REWARD_MODEL_RETRY_DELAY)
                    continue
            except Exception as e:
                warnings.warn(f"Unexpected error calling reward model (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(REWARD_MODEL_RETRY_DELAY)
                    continue
        
        # All retry attempts failed, return minimum score
        warnings.warn(f"All {max_retries} attempts to call reward model failed, returning {MIN_REWARD_SCORE}")
        return MIN_REWARD_SCORE
    
    def _call_reward_model_for_critique(self, question: str, answer: str, max_retries: int = 10) -> tuple[float, str, str]:
        """
        Call the reward model to get score, critique, and full response.
        
        Args:
            question: The question text
            answer: The answer text to evaluate
            max_retries: Maximum number of retry attempts
            
        Returns:
            tuple: (score, critique, full_response) - Score, critique text, and complete response
        """
        if not self.reward_model_url:
            warnings.warn("Reward model URL not configured, returning default values")
            return MIN_REWARD_SCORE, "", ""
        
        # Build complete chat completions URL
        chat_url = f"{self.reward_model_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # Build prompt using the REWARD_MODEL_PROMPT template
        system_prompt = REWARD_MODEL_PROMPT
        
        prompt = f"Question: {question}\n\nAgent Trajectory: {answer}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": MAX_TOKENS_WITH_CRITIQUE,
        }
        
        # Add model name to payload if specified in config
        reward_model_name = self.reward_model_config.get("model")
        if reward_model_name:
            payload["model"] = reward_model_name
        
        for attempt in range(max_retries):
            try:
                print(f"Calling reward model for critique (attempt {attempt + 1}/{max_retries})...", flush=True)
                http_response = requests.post(chat_url, headers=headers, json=payload, timeout=REWARD_MODEL_TIMEOUT)
                http_response.raise_for_status()
                
                result = http_response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                if not content:
                    print(f"Attempt {attempt + 1}: Empty response from reward model", flush=True)
                    if attempt < max_retries - 1:
                        time.sleep(REWARD_MODEL_RETRY_DELAY)
                        continue
                    return MIN_REWARD_SCORE, "", ""
                
                # 提取<score>标签中的内容
                score_match = re.search(r'<score>\s*([\d.]+)\s*</score>', content, re.DOTALL | re.IGNORECASE)
                score = MIN_REWARD_SCORE
                if score_match:
                    score_str = score_match.group(1).strip()
                    score = float(score_str)
                    if not (MIN_REWARD_SCORE <= score <= MAX_REWARD_SCORE):
                        warnings.warn(f"Reward model score {score} out of range [{MIN_REWARD_SCORE},{MAX_REWARD_SCORE}]")
                        score = max(MIN_REWARD_SCORE, min(MAX_REWARD_SCORE, score))
                
                # Extract critique from <critique> tag
                critique_match = re.search(r'<critique>(.*?)</critique>', content, re.DOTALL | re.IGNORECASE)
                critique = critique_match.group(1).strip() if critique_match else ""
                
                if score_match or critique_match:
                    print(f"Successfully got reward model response: score={score}, critique_length={len(critique)}", flush=True)
                    return score, critique, content
                else:
                    print(f"Attempt {attempt + 1}: No <score> or <critique> tag found in reward model response", flush=True)
                    print(f"Reward model response: {content[:200]}...", flush=True)
                    if attempt < max_retries - 1:
                        print("Retrying reward model call...", flush=True)
                        time.sleep(REWARD_MODEL_RETRY_DELAY)
                        continue
                        
            except requests.exceptions.RequestException as e:
                warnings.warn(f"Error calling reward model (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(REWARD_MODEL_RETRY_DELAY)
                    continue
            except Exception as e:
                warnings.warn(f"Unexpected error calling reward model (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(REWARD_MODEL_RETRY_DELAY)
                    continue
        
        # All retry attempts failed, return default values
        warnings.warn(f"All {max_retries} attempts to call reward model failed, returning default values")
        return MIN_REWARD_SCORE, "", ""

    def reset(self):
        """
        Reset the environment and return initial observations.
        
        Returns:
            tuple: (task, info_dict) - Task information and additional info
        """
        self.step_count = 0
        
        # If this is a critique round, preserve critique information
        if isinstance(self.task, dict) and self.task.get("is_critique_round"):
            # Keep task unchanged, only reset step count
            return self.task, {}
        
        return self.task, {}

    def step(self, action: list[dict] | str | dict):
        """
        Take a step in the environment based on the action.

        Args:
            actions: List containing a single action string from the agent

        Returns:
            next_observations, rewards, terminateds, infos
        """
        if action is None:
            action = []

        if isinstance(action, dict):
            action = [action]
        self.step_count += 1

        reward = 0
        # Check if we should terminate
        done = self.step_count >= self.max_steps or isinstance(action, str)
        # Check if action contains a "finish" tool call
        if isinstance(action, list) and action:
            for tool_call in action:
                if tool_call.get("function", {}).get("name") == "finish":
                    done = True
                    break
        if done:
            # Cannot find tool calls which means the agent is not using the tool and is done.
            if isinstance(action, str):
                llm_response = action
            elif isinstance(action, list):
                # Find the finish tool call
                finish_action = None
                for tool_call in action:
                    if tool_call.get("function", {}).get("name") == "finish":
                        finish_action = tool_call
                        break
                if finish_action:
                    arguments = finish_action.get("function", {}).get("arguments", {})
                    llm_response = arguments.get("response", "")
                else:
                    # No finish tool call found, use the action itself
                    llm_response = str(action)

            task_info = self.task if self.task is not None else {}
            # Calculate base reward
            reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            base_reward = reward_output.reward
            
            # If reward model is enabled, add reward model score
            if self.enable_reward_model:
                question = task_info.get("question", "")
                rm_score = self._call_reward_model(question, llm_response)
                
                # Calculate final reward: base_reward + lambda * rm_score
                final_reward = base_reward + self.reward_model_lambda * rm_score
                
                # Update metadata
                reward_output.metadata["reward_model_score"] = rm_score
                reward_output.metadata["base_reward"] = base_reward
                reward_output.metadata["reward_model_lambda"] = self.reward_model_lambda
                reward_output.metadata["final_reward"] = final_reward
                
                print(f"[ToolEnvironment] Base reward: {base_reward}, RM score: {rm_score}, Final reward: {final_reward}", flush=True)
            else:
                final_reward = base_reward
                # Record base reward even if reward model is not enabled
                reward_output.metadata["base_reward"] = base_reward
                reward_output.metadata["final_reward"] = final_reward
    
            return {}, final_reward, done, {
                "response": action, 
                "metadata": reward_output.metadata, 
                "is_correct": reward_output.is_correct
            }
            # reward_output = self.reward_fn(task_info=task_info, action=llm_response)
            # return {}, reward_output.reward, done, {"response": action, "metadata": reward_output.metadata, "is_correct": reward_output.is_correct}

        tool_calls = action
        assert isinstance(tool_calls, list)
        tool_outputs = self._execute_tool_calls(tool_calls)
        next_obs = {"tool_outputs": tool_outputs}

        # Return results as lists with single items to maintain batch structure
        return next_obs, reward, done, {"response": action, "metadata": {}}

    def _execute_tool_calls(self, tool_calls: list[dict[Any, Any]]) -> dict[str, str]:
        import threading

        # Create a dictionary to store results in order
        tool_outputs: dict[str, str] = {}
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        threads = []

        def execute_tool(tool_call):
            try:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_output = self.tools(tool_name=tool_name, **tool_args)
                tool_output_str = tool_output.to_string()

                output_queue.put((tool_call["id"], tool_output_str))
            except Exception as e:
                # Catch all exceptions to prevent thread crash from hanging main thread
                error_msg = f"Tool execution failed: {str(e)}"
                print(f"[ToolEnv] Thread error for tool {tool_call.get('function', {}).get('name', 'unknown')}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                output_queue.put((tool_call["id"], f"Error: {error_msg}"))
        # Create and start a thread for each tool call
        for idx, tool_call in enumerate(tool_calls):
            thread = threading.Thread(target=execute_tool, args=(tool_call,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=TOOL_EXECUTION_TIMEOUT)
            if thread.is_alive():
                print(f"[ToolEnv] Warning: Thread timed out after {TOOL_EXECUTION_TIMEOUT}s, killing it", flush=True)
      

        # Collect results and store in order
        while not output_queue.empty():
            tool_call_id, output_str = output_queue.get()
            tool_outputs[tool_call_id] = output_str

        return tool_outputs

    @staticmethod
    def from_dict(env_args: dict) -> "ToolEnvironment":
        """
        Create a ToolEnvironment instance from a dictionary of arguments.
        
        Args:
            env_args: Dictionary containing environment configuration
            
        Returns:
            ToolEnvironment: Initialized environment instance
        """
        tools = env_args.pop("tools", None)
        tool_map = env_args.pop("tool_map", None)
        reward_fn = env_args.pop("reward_fn", None)
        max_steps = env_args.pop("max_steps", 10)
        reward_model_config = env_args.pop("reward_model_config", None)
       
        return ToolEnvironment(task=env_args, 
                               tools=tools, 
                               tool_map=tool_map, 
                               max_steps=max_steps, 
                               reward_fn=reward_fn,
                               reward_model_config=reward_model_config)
