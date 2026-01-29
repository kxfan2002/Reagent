import dataclasses
from abc import abstractmethod
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput


@dataclasses.dataclass
class CodeToolOutput(ToolOutput):
    stdout: str | None = None  # Standard output
    stderr: str | None = None  # Standard error
    output: str | None = None  # Result of the code execution (i.e. last line of code)

    def to_string(self) -> str:
        """
        Convert the code tool output to a string representation.

        Returns:
            str: A string representation of the output, prioritizing stdout, then output, then stderr.
        """
        # Prioritize stdout (print statements) for better visibility
        if self.stdout:
            return str(self.stdout)
        # Then use the last expression result
        elif self.output is not None:
            return str(self.output)
        # Show errors if present
        elif self.stderr:
            return str(self.stderr)
        # If both stdout and output are empty and no error, indicate successful execution
        else:
            return "Execution completed successfully with no output."


class CodeTool(Tool):
    """Base class for Python code execution tools.

    This abstract class defines a common interface for all Python code execution tools,
    whether they run code locally or in a remote sandbox like E2B.
    """

    def __init__(self, name: str, description: str, n_sandboxes: int = 1):
        """Initialize the Python code tool.

        Args:
            name: The name of the tool
            description: Description of what the tool does
            n_sandboxes: Number of concurrent sandboxes/workers to use
        """
        self.n_sandboxes = n_sandboxes
        super().__init__(name=name, description=description)

    @property
    def json(self) -> dict[str, Any]:
        """Return the tool's information in the required format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute in the sandbox environment.",
                        }
                    },
                    "required": ["code"],
                },
            },
        }

    @abstractmethod
    def forward(self, code: str, timeout: int = 12, **kwargs) -> CodeToolOutput:
        """
        Execute Python code in the sandbox environment.

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds
            **kwargs: Additional parameters specific to the implementation

        Returns:
            Dict containing execution results, stdout, and stderr
        """
        pass

    def _init_sandbox(self):
        """Initialize the sandbox environment(s)."""
        pass

    def _kill_sandbox(self):
        """Clean up all sandbox resources."""
        pass

    def _restart_sandbox(self):
        """Restart the sandbox environment."""
        pass

    def __del__(self):
        """Cleanup when the interpreter is destroyed."""
        self._kill_sandbox()
