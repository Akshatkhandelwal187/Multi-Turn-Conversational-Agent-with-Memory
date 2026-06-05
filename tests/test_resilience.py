"""Tests for the retrying chat-model wrapper."""

from __future__ import annotations

import pytest

from aria.exceptions import ModelError
from aria.models.fakes import FlakyModel
from aria.models.resilience import RetryingChatModel


def test_retries_then_succeeds():
    inner = FlakyModel(fail_times=2, reply="recovered")
    model = RetryingChatModel(inner, max_retries=3, base_delay=0.001)
    out = model.invoke("hi")
    assert out.content == "recovered"
    assert inner.calls == 3  # failed twice, succeeded on the third


def test_raises_typed_error_after_exhaustion():
    model = RetryingChatModel(FlakyModel(fail_times=10), max_retries=2, base_delay=0.001)
    with pytest.raises(ModelError):
        model.invoke("hi")


def test_bind_tools_returns_wrapped_model():
    model = RetryingChatModel(FlakyModel(fail_times=0), max_retries=1, base_delay=0.001)
    bound = model.bind_tools([])
    assert isinstance(bound, RetryingChatModel)
