"""Offline tests for the pluggable quality scorers (VLM-judge parsing + factory)."""

from __future__ import annotations

import types

import pytest

from uni_agent.reward.quality_models import VLMJudgeScorer, get_quality_scorer

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False


class _FakeChatCompletions:
    def __init__(self, reply: str):
        self._reply = reply
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        msg = types.SimpleNamespace(content=self._reply)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, reply: str):
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions(reply))


def _png(tmp_path):
    p = tmp_path / "x.png"
    if _HAS_PIL:
        Image.new("RGB", (32, 32), (10, 20, 30)).save(p)
    else:  # pragma: no cover
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return p


@pytest.mark.parametrize(
    "reply,expected",
    [("8", (8 - 1) / 9.0), ("Rating: 10/10", 1.0), ("1", 0.0), ("garbage", 0.0), ("42", 1.0)],
)
def test_vlm_judge_parsing(tmp_path, reply, expected):
    img = _png(tmp_path)
    scorer = VLMJudgeScorer(model="fake-vlm", client=_FakeClient(reply))
    s = scorer.score(str(img), "a red bicycle at sunset")
    assert s == pytest.approx(expected)
    assert 0.0 <= s <= 1.0
    # the image was sent as an image_url content part
    content = scorer.client.chat.completions.last_call["messages"][0]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_factory_none_and_unknown():
    assert get_quality_scorer(None) is None
    assert get_quality_scorer("") is None
    with pytest.raises(ValueError):
        get_quality_scorer("does_not_exist")


def test_factory_caches_instances():
    a = get_quality_scorer({"name": "vlm_judge", "model": "m", "base_url": "http://x/v1"})
    b = get_quality_scorer({"name": "vlm_judge", "model": "m", "base_url": "http://x/v1"})
    assert a is b  # cached by config so torch-backed scorers load once
