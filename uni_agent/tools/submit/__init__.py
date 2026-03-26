"""Submit (finish) tool definition."""

from pathlib import Path

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
A simple submit tool to finish tasks.
This tool signals completion of a task or submission of results.
""".strip()


@register_tool("submit")
class SubmitTool(AbstractTool):
    @property
    def name(self) -> str:
        return "submit"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "submit"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "submit",
                "description": DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    def get_install_command(self) -> str:
        return None
