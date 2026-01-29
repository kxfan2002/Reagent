from typing import Protocol, runtime_checkable

from rllm.agents.agent import Action
from rllm.rewards.code_reward import RewardCodeFn
from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.mix_reward import RewardMixFn, mixed_reward_fn as _mixed_reward_fn, adaptive_reward_fn as _adaptive_reward_fn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


@runtime_checkable
class RewardFunction(Protocol):
    """Protocol for reward functions"""

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for an agent's action.

        Args:
            task_info: The task dictionary containing question, answer, and other metadata
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward value, either as a float or RewardOutput object
        """
        ...


# Simple example implementation
def zero_reward(task_info: dict, action: str) -> RewardOutput:
    """
    A simple reward function that always returns zero.
    Useful as a placeholder when no specific reward logic is needed.

    Args:
        task: The task dictionary
        action: The agent's response

    Returns:
        float: Always returns 0.0
    """
    return RewardOutput(reward=0.0, metadata={})


def math_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for math tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on math evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardMathFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def search_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for search tasks that implements the RewardFunction protocol.

    Args:
        task_info: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value based on search evaluation
    """
    reward_config = RewardConfig()
    reward_fn = RewardSearchFn(reward_config)
    if isinstance(action, Action):
        action = action.action

    # Create RewardInput from task_info and action
    reward_input = RewardInput(task_info=task_info, action=action)

    return reward_fn(reward_input)


def code_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    A reward function for code tasks that implements the RewardFunction protocol.

    Args:
        task: The task dictionary containing data_source, ground_truth and other metadata
        action: The agent's response/solution

    Returns:
        float: The calculated reward value based on code execution results
    """
    reward_config = RewardConfig()
    reward_fn = RewardCodeFn(reward_config)
    if isinstance(action, Action):
        action = action.action
    return reward_fn(task_info, action)


def mixed_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    Mixed reward function: automatically selects math or text evaluation based on answer type.
    
    - Math answers (integers, decimals, fractions, LaTeX expressions) → uses math evaluation
    - Text answers → uses text similarity evaluation (F1/EM)
    
    Args:
        task_info: Task information dictionary containing ground_truth and other metadata
        action: The model's response
        
    Returns:
        RewardOutput: Contains reward, is_correct, and metadata
    """
    if isinstance(action, Action):
        action = action.action
    return _mixed_reward_fn(task_info, action)


def adaptive_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    Adaptive reward function: automatically selects evaluation method based on data_source field.
    
    - data_source == "math" → forces math evaluation
    - data_source == "search" → forces text evaluation
    - other → automatically determines answer type
    
    Args:
        task_info: Task information dictionary, may contain data_source field
        action: The model's response
        
    Returns:
        RewardOutput: Contains reward, is_correct, and metadata
    """
    if isinstance(action, Action):
        action = action.action
    return _adaptive_reward_fn(task_info, action)
