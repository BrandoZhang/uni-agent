"""Image-generation tool definition.

A frozen scaffold tool that turns a text prompt into an image by shelling out to the
``media-ai`` CLI (see the sibling ``generate_image`` script). The default ``mock``
provider is fully offline, so the tool -- and the whole image agent -- runs with no
API keys, no network and no GPU; swap ``$MEDIA_PROVIDER`` for a real backend to get
real images with no code change.

This tool is the media-generation analog of DeepEyes' ``image_zoom_in_tool``: the
VLM chooses when to call it, and (with multimodal feedback enabled in
``AgentInteraction``) the generated image is fed back so the model can inspect and
refine it across turns.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool

DESCRIPTION = """
Generate an image from a text prompt. Write a vivid, detailed prompt describing the
subject, style, composition, lighting and colors. The tool returns the path, size and
metadata of the generated image. If image feedback is enabled you will then SEE the
image in the next turn and can decide to refine the prompt and call this tool again,
or call `finish` when you are satisfied.
""".strip()


class GenerateImageArguments(BaseModel):
    prompt: str = Field(description="Detailed text description of the image to generate.")
    size: str = Field(
        default="1024x1024",
        description="Image size as WxH in pixels, e.g. '1024x1024', '512x512', '1024x576'.",
    )
    negative_prompt: str | None = Field(
        default=None,
        description="Optional: content/attributes to avoid in the image.",
    )
    seed: int | None = Field(
        default=None,
        description="Optional: integer seed for reproducible generation.",
    )


@register_tool("generate_image")
class GenerateImageTool(AbstractTool):
    @property
    def name(self) -> str:
        return "generate_image"

    @property
    def local_path(self) -> Path:
        return Path(__file__).parent / "generate_image"

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(
            description=DESCRIPTION,
            arguments_model=GenerateImageArguments,
        )

    def get_install_command(self) -> str | None:
        # media-ai is an external dependency; verify it is importable or on PATH.
        return (
            'python -c "import media_ai" >/dev/null 2>&1 || media-ai --help >/dev/null 2>&1 || '
            "( echo 'media-ai not found. Install with: pip install -e /path/to/media-ai' >&2; exit 1 )"
        )
