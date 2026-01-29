from rllm.tools.code_tools import (
    PythonInterpreter,
)
from rllm.tools.registry import ToolRegistry
from rllm.tools.web_tools import (
    # FirecrawlTool,
    # GoogleSearchTool,
    # TavilyExtractTool,
    # TavilySearchTool,
    WebSearchTool,
    JinaBrowseTool
)

from rllm.tools.file_tools import(
    FileReaderTool
)

from rllm.tools.multimodal_tools import(
    Audio2TextTool,
    Image2TextTool
)
# Define default tools dict
DEFAULT_TOOLS = {
    "python": PythonInterpreter,
    # "google_search": GoogleSearchTool,
    # "firecrawl": FirecrawlTool,
    # "tavily-extract": TavilyExtractTool,
    # "tavily-search": TavilySearchTool,
    "search": WebSearchTool,
    "browse": JinaBrowseTool,
    "image2text": Image2TextTool,
    "audio2text": Audio2TextTool,
    "file_reader": FileReaderTool
}

# Create the singleton registry instance and register all default tools
tool_registry = ToolRegistry()
tool_registry.register_all(DEFAULT_TOOLS)

# __all__ = ["PythonInterpreter", "LocalRetrievalTool", "GoogleSearchTool", "FirecrawlTool", "TavilyExtractTool", "TavilySearchTool", "ToolRegistry", "tool_registry"]
__all__ = ["PythonInterpreter", "WebSearchTool", "JinaBrowseTool", "Image2TextTool", "Audio2TextTool", "FileReaderTool", "ToolRegistry", "tool_registry"]
