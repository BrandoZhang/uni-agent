# ruff: noqa
"""
Scaffold tools.
"""

from pydantic import BaseModel

from .finish import FinishTool
from .registry import get_tool, AbstractTool
from .execute_bash import ExecuteBashTool
from .lark_cli import LarkCliTool
from .media_gen import (
    ConcatVideoTool,
    Image2ImageTool,
    Image2VideoTool,
    MediaUsageTool,
    Ref2VideoTool,
    Text2ImageTool,
    Text2VideoTool,
    VideoTaskTool,
)
from .search_arxiv import SearchArxivTool
from .search import SearchWikiTool
from .str_replace_editor import StrReplaceEditorTool
from .submit import SubmitTool


class ToolConfig(BaseModel):
    name: str

    def get_tool(self) -> AbstractTool:
        """Return a tool instance (for env.install_tools / init_for_interaction)."""
        return get_tool(self.name)


__all__ = [
    "ToolConfig",
    "ExecuteBashTool",
    "FinishTool",
    "LarkCliTool",
    "Text2ImageTool",
    "Image2ImageTool",
    "Text2VideoTool",
    "Image2VideoTool",
    "Ref2VideoTool",
    "ConcatVideoTool",
    "VideoTaskTool",
    "MediaUsageTool",
    "SearchArxivTool",
    "SearchWikiTool",
    "StrReplaceEditorTool",
    "SubmitTool",
]
