"""
Dynamic prompt generator based on dataset configuration.
"""
import json
import os
from typing import List

# Tool definitions with their full descriptions
TOOL_DEFINITIONS = {
    "python": {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Execute Python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code snippet to execute. You need to use 'print' function to print the result."
                    }
                },
                "required": ["code"]
            }
        }
    },
    "search": {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search tool that performs batched web searches: supply an array 'queries'; the tool retrieves search results for each query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
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
    },
    "browse": {
        "type": "function",
        "function": {
            "name": "browse",
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
    },
    "image2text": {
        "type": "function",
        "function": {
            "name": "image2text",
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
    },
    "audio2text": {
        "type": "function",
        "function": {
            "name": "audio2text",
            "description": "Convert audio into text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "The local file path or URL of the audio file to be transcribed."
                    }
                },
                "required": ["audio_path"]
            }
        }
    },
    "file_reader": {
        "type": "function",
        "function": {
            "name": "file_reader",
            "description": "Read and extract content from various file formats including xml, xlsx, docs, csv, pdf, txt, json, jsonl, jsonld, pptx, py, zip, pdb. You are not allowed to change the original file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be read. Supports ZIP internal files with format: zip_path::internal_file_path"
                    }
                },
                "required": ["file_path"]
            }
        }
    }
}


def generate_system_prompt(tool_list: List[str]) -> str:
    """
    Generate system prompt based on the list of available tools.
    
    Args:
        tool_list: List of tool names to include in the prompt
        
    Returns:
        Complete system prompt string
    """
    import json
    
    # Filter tool definitions based on tool_list
    tools_json = []
    for tool_name in tool_list:
        if tool_name in TOOL_DEFINITIONS:
            tools_json.append(json.dumps(TOOL_DEFINITIONS[tool_name], ensure_ascii=False))
    
    tools_section = "\n".join(tools_json)
    
    prompt = f"""You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_section}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
You should first analysis the question and then answer it step by step. Divide your problem solving into small tasks. Don't search the entire question directly. Don't repeat searching the same query."""
    
    return prompt


def load_dataset_config(config_path: str = None) -> dict:
    """
    Load dataset configuration from JSON file.
    
    Args:
        config_path: Path to the configuration JSON file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Try JSON first, then YAML for backward compatibility
        json_path = os.path.join(os.path.dirname(__file__), "dataset_config.json")
        yaml_path = os.path.join(os.path.dirname(__file__), "dataset_config.yaml")
        
        if os.path.exists(json_path):
            config_path = json_path
        elif os.path.exists(yaml_path):
            config_path = yaml_path
        else:
            raise FileNotFoundError("No configuration file found (dataset_config.json or dataset_config.yaml)")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.json'):
            config = json.load(f)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config = yaml.safe_load(f)
        else:
            # Try JSON first
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                import yaml
                f.seek(0)
                config = yaml.safe_load(f)
    
    return config


def get_dataset_tools(dataset_name: str, config_path: str = None) -> List[str]:
    """
    Get the list of tools for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        config_path: Path to the configuration YAML file (optional)
        
    Returns:
        List of tool names
    """
    config = load_dataset_config(config_path)
    
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration. "
                        f"Available datasets: {list(config['datasets'].keys())}")
    
    return config['datasets'][dataset_name]['tools']


def get_dataset_config(dataset_name: str, config_path: str = None) -> dict:
    """
    Get the complete configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset
        config_path: Path to the configuration YAML file (optional)
        
    Returns:
        Dataset configuration dictionary
    """
    config = load_dataset_config(config_path)
    
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration. "
                        f"Available datasets: {list(config['datasets'].keys())}")
    
    # Merge with defaults
    dataset_config = config['datasets'][dataset_name].copy()
    defaults = config.get('defaults', {})
    
    # Add defaults for missing keys
    for key, value in defaults.items():
        if key not in dataset_config:
            dataset_config[key] = value
    
    return dataset_config


if __name__ == "__main__":
    # Test the prompt generator
    import sys
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        dataset_name = "gaia"
    
    try:
        tools = get_dataset_tools(dataset_name)
        print(f"Dataset: {dataset_name}")
        print(f"Tools: {tools}")
        print("\n" + "="*80)
        print("Generated System Prompt:")
        print("="*80)
        print(generate_system_prompt(tools))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

