from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple


class BaseTool(ABC):
    """Base class for all tools.
    
    Each tool should implement:
    - extract_tool_call: Extract the tool call content from model response, return None if not match
    - execute: Execute the tool with extracted content
    - release: Clean up resources when done
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the tool with configuration.
        
        Args:
            config: Configuration dictionary for this tool
        """
        self.config = config
    
    @abstractmethod
    def extract_tool_call(self, content: str) -> Any:
        """Extract tool call from response. Return None if not match."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, tool_content: Any) -> Tuple[str, bool]:
        """Execute tool. Returns (result_str, success_flag)."""
        raise NotImplementedError
        
    def release(self):
        """Release resources held by this tool.
        
        Override this method if the tool needs cleanup.
        """
        pass
