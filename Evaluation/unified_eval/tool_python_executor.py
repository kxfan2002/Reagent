import io
import sys
import traceback
import multiprocessing
import queue
import signal
from typing import Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool


class TimeoutException(Exception):
    """Exception raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for execution timeout."""
    raise TimeoutException("Code execution timed out")


def _execute_in_subprocess(code: str, result_queue: multiprocessing.Queue, timeout: int):
    """
    Execute Python code in a subprocess with output capture.
    
    This function runs in a separate process to ensure complete isolation
    from the parent process and thread-safe output capturing.
    
    Args:
        code: Python code to execute
        result_queue: Queue to pass results back to parent process
        timeout: Maximum execution time in seconds
    """
    # Create StringIO objects to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Set up timeout signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Redirect stdout/stderr (safe in subprocess)
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        
        # Create a namespace for code execution
        namespace = {
            '__builtins__': __builtins__,
        }
        
        # Execute the code
        try:
            # Try to compile as eval (single expression)
            compiled = compile(code, '<string>', 'eval')
            result = eval(compiled, namespace)
            if result is not None:
                print(result)
        except SyntaxError:
            # If it's not a single expression, execute as statements
            exec(code, namespace)
        
        # Cancel the alarm
        signal.alarm(0)
        
        # Get captured output
        output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        
        if stderr_output:
            output += f"\nSTDERR:\n{stderr_output}"
        
        result_msg = output if output else "Code executed successfully (no output)"
        result_queue.put(("success", result_msg))
        
    except TimeoutException:
        signal.alarm(0)
        result_queue.put(("error", "Execution Error:\nTimeoutException: Code execution timed out"))
        
    except Exception as e:
        signal.alarm(0)
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        result_queue.put(("error", f"Execution Error:\n{error_msg}"))
        
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close buffers
        stdout_buffer.close()
        stderr_buffer.close()


def execute_python_code(code: str, timeout: int = 30) -> str:
    """
    Execute Python code and return the result (thread-safe, using subprocess).
    
    This function uses multiprocessing to execute code in a completely isolated
    subprocess, ensuring thread safety and preventing output interference between
    concurrent executions.
    
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)
    
    Returns:
        String containing the output or error message
    """
    # Use multiprocessing to isolate code execution in a separate process
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    
    # Create and start the process
    p = multiprocessing.Process(
        target=_execute_in_subprocess,
        args=(code, result_queue, timeout),
    )
    p.start()
    
    # Wait for the process to complete with additional buffer time
    p.join(timeout=timeout + 2)
    
    try:
        # Get the result from the queue (with a short timeout)
        status, message = result_queue.get(timeout=1)
        return message
    except queue.Empty:
        # Return timeout message if no result is available
        return "Execution Error:\nTimeoutException: Code execution timed out (no response from subprocess)"
    finally:
        # Ensure the process is terminated if still running
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
            if p.is_alive():
                p.kill()
                p.join()


@register_tool("python", allow_overwrite=True)
class PythonExecutor(BaseTool):
    """Tool for executing Python code."""
    
    name = "python"
    description = "Execute Python code and return the result. Use this tool to perform calculations, data processing, or any computational tasks."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. The code will be executed and the result will be returned."
            }
        },
        "required": ["code"]
    }
    
    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.max_output_length = cfg.get("max_output_length", 10000) if cfg else 10000
        self.timeout = cfg.get("timeout", 30) if cfg else 30
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        """Execute Python code and return formatted result."""
        try:
            code = params["code"]
        except:
            return "[Python] Invalid request format: Input must be a JSON object containing 'code' field"
        
        if not code or not isinstance(code, str):
            return "[Python] Error: 'code' is missing, empty, or not a string"
        
        try:
            # Get timeout from kwargs or use default
            timeout = kwargs.get("timeout", self.timeout)
            result = execute_python_code(code, timeout=timeout)
            
            # Truncate if too long
            if len(result) > self.max_output_length:
                result = result[:self.max_output_length] + f"\n... (output truncated, total length: {len(result)} chars)"
            
            return result
            
        except Exception as e:
            return f"[Python] Error: {str(e)}"


if __name__ == "__main__":
    # Test the tool
    executor = PythonExecutor()
    
    # Test 1: Simple calculation
    print("Test 1: Simple calculation")
    result = executor.call({"code": "2 + 2"})
    print(result)
    print()
    
    # Test 2: Print statement
    print("Test 2: Print statement")
    result = executor.call({"code": "print('Hello, World!')\nprint('This is a test')"})
    print(result)
    print()
    
    # Test 3: Complex calculation
    print("Test 3: Complex calculation")
    result = executor.call({"code": """
import math
result = math.sqrt(16) + math.pi
print(f"The result is: {result}")
result
"""})
    print(result)
    print()
    
    # Test 4: Error handling
    print("Test 4: Error handling")
    result = executor.call({"code": "1 / 0"})
    print(result)
    print()
    
    # Test 5: User's example
    print("Test 5: User's calculation example")
    result = executor.call({"code": """# Calculate the number of incorrect papers
articles = 1002
average_p_value = 0.04

# Assuming p-value represents the false positive rate
incorrect_papers = articles * average_p_value
rounded_up = int(incorrect_papers) + 1 if incorrect_papers % 1 > 0 else int(incorrect_papers)

print(f"Total articles: {articles}")
print(f"Average p-value: {average_p_value}")
print(f"Calculated incorrect papers: {incorrect_papers}")
print(f"Rounded up to next integer: {rounded_up}")"""})
    print(result)
