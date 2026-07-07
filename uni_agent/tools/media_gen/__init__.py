"""Multimodal *creation* tools: text/image/video/audio -> image/video.

These are thin uni-agent registrations for the standalone **media-ai** CLI
toolkit (``packages/media-ai``). The actual implementation + HTTP calls live
in that pip-installable package (``pip install -e packages/media-ai``); here we
only declare the tool schemas so the model can call the commands.

Because the commands ship as ``media-ai`` console scripts (``text2image``,
``image2video``, ...), these are registered as **system tools**
(``copy_to_remote=False``): the runtime is expected to have ``media-ai``
installed (so the scripts are on PATH), rather than uni-agent copying a script
in. This keeps the toolkit decoupled and reusable across agent frameworks.

Generation tools: text2image, image2image, text2video, image2video, ref2video,
concat_video. Utility tools: video_task, media_usage. Default backend is an
offline mock; set ``MEDIA_BACKEND=volc`` + ``ARK_API_KEY`` for the real Ark API.
"""

from pydantic import BaseModel, Field

from uni_agent.tools.base import AbstractTool
from uni_agent.tools.registry import register_tool


# --------------------------------------------------------------------------
# argument models
# --------------------------------------------------------------------------


class Text2ImageArgs(BaseModel):
    prompt: str = Field(description="Text description of the image to generate.")
    output: str = Field(description="Path to write the generated image, e.g. workspace/ref_hero.png")
    width: int = Field(default=768, description="Image width in pixels.")
    height: int = Field(default=432, description="Image height in pixels.")
    max_images: int = Field(default=1, description="If >1, generate a related group of images (saved as output, output_2, ...).")
    seed: int | None = Field(default=None, description="Optional seed for reproducibility.")
    backend: str | None = Field(default=None, description="mock (default, offline) or volc (real API).")


class Image2ImageArgs(BaseModel):
    images: list[str] = Field(description="Reference image path(s) to derive from (1..14).")
    prompt: str = Field(description="How to transform the reference / what to render.")
    output: str = Field(description="Path to write the new image.")
    strength: float = Field(default=0.6, description="How much to follow the prompt vs the reference (0-1).")
    max_images: int = Field(default=1, description="If >1, generate a related group.")
    seed: int | None = Field(default=None, description="Optional seed for reproducibility.")
    backend: str | None = Field(default=None, description="mock (default) or volc (real API).")


class Text2VideoArgs(BaseModel):
    prompt: str = Field(description="What happens in this shot.")
    output: str = Field(description="Path to write the .mp4, e.g. workspace/shot1.mp4")
    seconds: int = Field(default=5, description="Clip length in seconds.")
    resolution: str = Field(default="480p", description="480p|720p|1080p. Lower is cheaper.")
    ratio: str = Field(default="16:9", description="16:9|9:16|1:1|4:3|3:4|21:9.")
    seed: int | None = Field(default=None, description="Optional seed.")
    camera_fixed: bool = Field(default=False, description="Fix the camera.")
    watermark: bool = Field(default=False, description="Add an AI watermark.")
    generate_audio: bool | None = Field(default=None, description="Generate synced audio (Seedance 2.0/1.5).")
    backend: str | None = Field(default=None, description="mock (default) or volc (real API).")


class Image2VideoArgs(BaseModel):
    first_frame: str = Field(description="Path to the first-frame / reference image (locks cross-shot consistency).")
    output: str = Field(description="Path to write the .mp4.")
    last_frame: str | None = Field(default=None, description="Optional last-frame image (first+last-frame interpolation).")
    prompt: str = Field(default="", description="The motion / action for the shot.")
    seconds: int = Field(default=5, description="Clip length in seconds.")
    resolution: str = Field(default="480p", description="480p|720p|1080p.")
    ratio: str = Field(default="adaptive", description="Aspect ratio; adaptive matches the first frame.")
    seed: int | None = Field(default=None, description="Optional seed.")
    camera_fixed: bool = Field(default=False, description="Fix the camera.")
    watermark: bool = Field(default=False, description="Add an AI watermark.")
    generate_audio: bool | None = Field(default=None, description="Generate synced audio.")
    return_last_frame: bool = Field(default=False, description="Also return the clip's last frame (to chain the next shot).")
    backend: str | None = Field(default=None, description="mock (default) or volc (real API).")


class Ref2VideoArgs(BaseModel):
    output: str = Field(description="Path to write the .mp4.")
    images: list[str] = Field(default_factory=list, description="Reference image path(s), 0-9 (role reference_image).")
    videos: list[str] = Field(default_factory=list, description="Reference video URL(s)/path(s), 0-3 (role reference_video).")
    audios: list[str] = Field(default_factory=list, description="Reference audio URL(s)/path(s), 0-3 (role reference_audio).")
    prompt: str = Field(default="", description="Text guidance.")
    seconds: int = Field(default=5, description="Clip length in seconds.")
    resolution: str = Field(default="480p", description="480p|720p|1080p.")
    ratio: str = Field(default="adaptive", description="Aspect ratio.")
    seed: int | None = Field(default=None, description="Optional seed.")
    watermark: bool = Field(default=False, description="Add an AI watermark.")
    generate_audio: bool | None = Field(default=None, description="Generate synced audio.")
    backend: str | None = Field(default=None, description="mock (default) or volc (real API).")


class ConcatVideoArgs(BaseModel):
    inputs: list[str] = Field(description="Ordered list of clip paths to join into the final film.")
    output: str = Field(description="Path to write the final .mp4.")
    width: int = Field(default=768, description="Output width in pixels.")
    height: int = Field(default=432, description="Output height in pixels.")


class VideoTaskArgs(BaseModel):
    op: str = Field(description="query | cancel.")
    id: str = Field(description="The async video task id.")
    backend: str | None = Field(default=None, description="mock or volc.")


class MediaUsageArgs(BaseModel):
    log: str | None = Field(default=None, description="Ledger path; defaults to $MEDIA_USAGE_LOG.")


# --------------------------------------------------------------------------
# tool registrations
# --------------------------------------------------------------------------


class _MediaTool(AbstractTool):
    """Base for the media tools: each maps to a ``media-ai`` console script.

    ``copy_to_remote=False`` -> the runtime must already have ``media-ai`` on
    PATH (``pip install -e packages/media-ai``); ``install_tools`` only does a
    ``which <name>`` presence check.
    """

    copy_to_remote = False

    _name: str
    _description: str
    _args: type[BaseModel]

    @property
    def name(self) -> str:
        return self._name

    def get_tool_schema(self) -> dict:
        return self.build_tool_schema(description=self._description, arguments_model=self._args)

    def get_install_command(self) -> str | None:
        # media-ai is provisioned out of band; nothing to install per-tool.
        return None


@register_tool("text2image")
class Text2ImageTool(_MediaTool):
    _name = "text2image"
    _description = "Generate an image from a text prompt (set max_images>1 for a related group). Use it to create reference assets that keep characters/style consistent across shots."
    _args = Text2ImageArgs


@register_tool("image2image")
class Image2ImageTool(_MediaTool):
    _name = "image2image"
    _description = "Generate a new image from one or more reference images + a prompt. Use it to derive consistent variants of a locked reference, or fuse multiple references."
    _args = Image2ImageArgs


@register_tool("text2video")
class Text2VideoTool(_MediaTool):
    _name = "text2video"
    _description = "Generate a short video clip from a text prompt. One call renders one shot. Good for scenery/establishing shots with no recurring subject."
    _args = Text2VideoArgs


@register_tool("image2video")
class Image2VideoTool(_MediaTool):
    _name = "image2video"
    _description = "Generate a clip from a first-frame image (optionally plus a last-frame). Prefer this for shots that must stay consistent with a reference asset; set return_last_frame to chain shots."
    _args = Image2VideoArgs


@register_tool("ref2video")
class Ref2VideoTool(_MediaTool):
    _name = "ref2video"
    _description = "Multimodal-reference video generation: mix reference images (0-9), videos (0-3), audio (0-3) and an optional prompt into one clip (Seedance 2.0). For asset-driven consistency, video editing/extension, or audio-synced shots."
    _args = Ref2VideoArgs


@register_tool("concat_video")
class ConcatVideoTool(_MediaTool):
    _name = "concat_video"
    _description = "Join per-shot video clips, in order, into one final film."
    _args = ConcatVideoArgs


@register_tool("video_task")
class VideoTaskTool(_MediaTool):
    _name = "video_task"
    _description = "Query or cancel an async video-generation task by id (cancelling a queued task cuts cost)."
    _args = VideoTaskArgs


@register_tool("media_usage")
class MediaUsageTool(_MediaTool):
    _name = "media_usage"
    _description = "Report accumulated generation cost (tokens, images, video seconds) from the usage ledger. Call it to summarize the run's cost."
    _args = MediaUsageArgs


__all__ = [
    "Text2ImageTool",
    "Image2ImageTool",
    "Text2VideoTool",
    "Image2VideoTool",
    "Ref2VideoTool",
    "ConcatVideoTool",
    "VideoTaskTool",
    "MediaUsageTool",
]
