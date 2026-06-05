"""Tests for the conversation registry and a UI import smoke test."""

from __future__ import annotations

import pytest

from aria.ui.conversations import ConversationRegistry


def test_create_and_list_orders_by_recency():
    reg = ConversationRegistry()
    a = reg.create("First")
    b = reg.create("Second")
    listed = reg.list()
    assert {c["id"] for c in listed} == {a, b}
    assert listed[0]["id"] == b  # most recently created first
    reg.touch(a)
    assert reg.list()[0]["id"] == a  # touching moves it to the top


def test_rename_and_delete():
    reg = ConversationRegistry()
    tid = reg.create("Old name")
    reg.rename(tid, "New name")
    assert reg.name_of(tid) == "New name"
    reg.delete(tid)
    assert len(reg) == 0


def test_persistence_round_trip(tmp_path):
    path = tmp_path / "conversations.json"
    reg = ConversationRegistry(path)
    tid = reg.create("Persisted chat")
    reopened = ConversationRegistry(path)
    assert reopened.name_of(tid) == "Persisted chat"


def test_streamlit_app_imports():
    pytest.importorskip("streamlit")
    from aria.ui import streamlit_app

    assert callable(streamlit_app.main)
