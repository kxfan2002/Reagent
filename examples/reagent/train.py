"""
Multi-tool agent training script with reinforcement learning.

This script trains an agent that can use multiple tools (Python, web search, 
file reading, image/audio processing) to solve complex tasks. The agent is 
trained using PPO (Proximal Policy Optimization) with GRPO variant.

Usage:
    python -m examples.reagent.train [hydra_config_overrides]
    
Example:
    python -m examples.reagent.train trainer.n_gpus_per_node=8 trainer.total_epochs=3
"""

import hydra

from rllm.agents import ToolAgent
from rllm.data.dataset import DatasetRegistry
from rllm.environments.tools.tool_env import ToolEnvironment
from rllm.rewards.reward_fn import adaptive_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer

# Configuration constants
AVAILABLE_TOOLS = ["python", "search", "browse", "audio2text", "image2text", "file_reader"]
PARSER_NAME = "qwen"  # Parser for tool calls

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful agent assistant that can solve problems with tools. \
You can use multiple tools multiple times, but you should only use them when necessary.

Tool instructions:
- search: Search the Internet to find relevant information or URLs, without reading webpage content.
- browse: Read and analyze the content of a given webpage URL to answer questions; web access only, no local resources.
- file_reader: Read local text-based files only; cannot access web pages (http), images, or audio.
- image2text: Analyze and answer questions about local images (jpg, png, etc.) only.
- audio2text: Transcribe local audio files (wav, mp3, etc.) into text only.
- python: Execute Python code for computation or data processing.

You should put your final answer in \\boxed{answer}."""


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    """
    Main training function using Hydra for configuration management.
    
    Args:
        config: Hydra configuration object containing all training parameters
    """
    # Load datasets registered during data preparation
    print("Loading datasets...")
    train_dataset = DatasetRegistry.load_dataset("reagent", "train")
    test_dataset = DatasetRegistry.load_dataset("aime2024", "test")
    print(f"Training examples: {len(train_dataset)}, Test examples: {len(test_dataset)}")

    # Configure agent with tools and system prompt
    agent_args = {
        "tools": AVAILABLE_TOOLS,
        "parser_name": PARSER_NAME,
        "system_prompt": SYSTEM_PROMPT,
    }
    
    # Configure reward model (optional external reward scoring)
    reward_model_config = {
        "enable": config.rllm.get("reward_model", {}).get("enable", False),
        "url": config.rllm.get("reward_model", {}).get("url", None),
        "lambda_weight": config.rllm.get("reward_model", {}).get("lambda", 0.3),  # Default weight for reward model
        "model": config.rllm.get("reward_model", {}).get("model", None),
    }
    
    # Configure environment with tools and reward function
    env_args = {
        "tools": AVAILABLE_TOOLS,
        "reward_fn": adaptive_reward_fn,  # Adaptive reward function that adjusts based on task type
        "reward_model_config": reward_model_config,
    }
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=ToolEnvironment,
        agent_args=agent_args,
        env_args=env_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")


if __name__ == "__main__":
    main()
