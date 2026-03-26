"""Execute bash command tool."""

from pathlib import Path

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Execute a bash command in the terminal.
""".strip()


@register_tool("execute_bash")
class ExecuteBashTool(AbstractTool):
    @property
    def name(self) -> str:
        return "execute_bash"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "execute_bash"

    def get_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": DESCRIPTION,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }

    def get_install_command(self) -> str:
        return None
