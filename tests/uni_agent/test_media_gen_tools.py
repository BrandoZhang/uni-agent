"""Schema guards for the media_gen tool registrations (uni-agent side).

These are thin declarations, but two things are easy to regress and matter for
the model's tool-call surface: every generation tool exposes an optional
``--model`` (Ark Model ID), and every async video tool exposes ``wait``. A
past bug dropped both from ``ref2video`` — this pins them.
"""

from __future__ import annotations

import pytest

from uni_agent.tools import media_gen

GENERATION_ARGS = [
    media_gen.Text2ImageArgs,
    media_gen.Image2ImageArgs,
    media_gen.Text2VideoArgs,
    media_gen.Image2VideoArgs,
    media_gen.Ref2VideoArgs,
]
VIDEO_ARGS = [media_gen.Text2VideoArgs, media_gen.Image2VideoArgs, media_gen.Ref2VideoArgs]


@pytest.mark.parametrize("args_model", GENERATION_ARGS)
def test_generation_tools_expose_optional_model(args_model):
    field = args_model.model_fields.get("model")
    assert field is not None, f"{args_model.__name__} is missing the --model field"
    assert field.default is None  # optional; falls back to CLI default


@pytest.mark.parametrize("args_model", VIDEO_ARGS)
def test_video_tools_expose_wait(args_model):
    field = args_model.model_fields.get("wait")
    assert field is not None, f"{args_model.__name__} is missing the wait field"
    assert field.default is True  # default behavior: block until the clip is ready


def test_ref2video_has_both_wait_and_model():
    # regression: Ref2VideoArgs previously lacked both
    fields = media_gen.Ref2VideoArgs.model_fields
    assert "wait" in fields and "model" in fields


def test_expected_tools_registered_and_batch_video_retired():
    exported = set(media_gen.__all__)
    assert exported == {
        "Text2ImageTool",
        "Image2ImageTool",
        "Text2VideoTool",
        "Image2VideoTool",
        "Ref2VideoTool",
        "ConcatVideoTool",
        "VideoTaskTool",
        "MediaUsageTool",
    }
    # batch_video was retired (parallelism belongs to the harness, not a tool)
    assert not hasattr(media_gen, "BatchVideoTool")
    assert not hasattr(media_gen, "BatchVideoArgs")


def test_media_tools_are_system_tools():
    # copy_to_remote=False -> media-ai must be on PATH; nothing is copied in
    assert media_gen.Text2ImageTool.copy_to_remote is False
