"""Finish tool definition."""

from pathlib import Path

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Finish the task and output the final answer.
Always call this tool when you are ready to end the interaction.
""".strip()


@register_tool("finish")
class FinishTool(AbstractTool):
    @property
    def name(self) -> str:
        return "finish"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "finish"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "finish",
                "description": DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "Final answer to return to the user.",
                        }
                    },
                    "required": ["answer"],
                },
            },
        }

    def get_install_command(self) -> str | None:
        return None
