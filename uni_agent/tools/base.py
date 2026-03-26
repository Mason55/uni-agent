"""Abstract base class for scaffold tools."""

from abc import ABC, abstractmethod


class AbstractTool(ABC):
    """Abstract tool definition with description and install command."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (e.g. execute_bash, str_replace_editor, submit)."""
        ...

    @abstractmethod
    def get_tool_schema(self) -> dict:
        """
        OpenAI tool schema: { \"type\": \"function\", \"function\": { ... } }.
        """
        ...

    @abstractmethod
    def get_install_command(self) -> str | None:
        """Command to run in container to complete tool installation. Return None if no extra install step."""
        ...
