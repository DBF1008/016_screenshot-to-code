"""Tests for turn-level file state rollback in AgentEngine."""

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from agent.engine import AgentEngine
from agent.providers.base import (
    EventSink,
    ExecutedToolCall,
    ProviderTurn,
)
from agent.tools.types import ToolCall


class FakeProviderSession:
    """A mock ProviderSession that yields pre-configured turns."""

    def __init__(self, turns: List[ProviderTurn]):
        self._turns = list(turns)
        self._index = 0
        self.appended_results: List[List[ExecutedToolCall]] = []

    async def stream_turn(self, on_event: EventSink) -> ProviderTurn:
        turn = self._turns[self._index]
        self._index += 1
        return turn

    def append_tool_results(
        self,
        turn: ProviderTurn,
        executed_tool_calls: list[ExecutedToolCall],
    ) -> None:
        self.appended_results.append(executed_tool_calls)

    async def close(self) -> None:
        pass


def _make_engine(
    initial_content: str = "",
    initial_path: str = "index.html",
) -> tuple[AgentEngine, AsyncMock]:
    mock_send: AsyncMock = AsyncMock()
    engine = AgentEngine(
        send_message=mock_send,
        variant_index=0,
        openai_api_key=None,
        openai_base_url=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        should_generate_images=False,
        initial_file_state=(
            {"path": initial_path, "content": initial_content}
            if initial_content
            else None
        ),
    )
    return engine, mock_send


def _extract_set_code_values(mock_send: AsyncMock) -> List[str]:
    """Extract all 'setCode' values from mock send_message calls."""
    values: List[str] = []
    for call in mock_send.call_args_list:
        args = call[0]
        msg_type = args[0]
        value = args[1]
        if msg_type == "setCode" and value is not None:
            values.append(value)
    return values


@pytest.mark.asyncio
async def test_rollback_on_second_edit_file_failure() -> None:
    """When the second edit_file fails, all file changes are rolled back."""
    original = "<div>hello</div>\n<p>world</p>"
    engine, mock_send = _make_engine(initial_content=original)

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="edit_file",
                arguments={"old_text": "hello", "new_text": "goodbye"},
            ),
            ToolCall(
                id="call-2",
                name="edit_file",
                arguments={"old_text": "NONEXISTENT", "new_text": "fail"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    assert engine.file_state.content == original

    set_code_values = _extract_set_code_values(mock_send)
    assert set_code_values[-1] == original


@pytest.mark.asyncio
async def test_no_rollback_when_all_edits_succeed() -> None:
    """When all edit_file calls succeed, the final state is preserved."""
    original = "<div>aaa</div>\n<p>bbb</p>"
    engine, mock_send = _make_engine(initial_content=original)

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="edit_file",
                arguments={"old_text": "aaa", "new_text": "AAA"},
            ),
            ToolCall(
                id="call-2",
                name="edit_file",
                arguments={"old_text": "bbb", "new_text": "BBB"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    expected = "<div>AAA</div>\n<p>BBB</p>"
    assert engine.file_state.content == expected


@pytest.mark.asyncio
async def test_all_tool_results_reported_to_model() -> None:
    """Both success and failure tool results are fed back to the model."""
    original = "<div>hello</div>"
    engine, _ = _make_engine(initial_content=original)

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="edit_file",
                arguments={"old_text": "hello", "new_text": "goodbye"},
            ),
            ToolCall(
                id="call-2",
                name="edit_file",
                arguments={"old_text": "NONEXISTENT", "new_text": "fail"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    assert len(session.appended_results) == 1
    executed = session.appended_results[0]
    assert len(executed) == 2
    assert executed[0].result.ok is True
    assert executed[1].result.ok is False


@pytest.mark.asyncio
async def test_rollback_with_three_edits_middle_fails() -> None:
    """With three edits where the second fails, all changes are rolled back."""
    original = "<h1>title</h1>\n<p>para</p>\n<span>note</span>"
    engine, _ = _make_engine(initial_content=original)

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="edit_file",
                arguments={"old_text": "title", "new_text": "TITLE"},
            ),
            ToolCall(
                id="call-2",
                name="edit_file",
                arguments={"old_text": "NONEXISTENT", "new_text": "fail"},
            ),
            ToolCall(
                id="call-3",
                name="edit_file",
                arguments={"old_text": "note", "new_text": "NOTE"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    assert engine.file_state.content == original


@pytest.mark.asyncio
async def test_rollback_restores_path_after_create_then_failing_edit() -> None:
    """If create_file changed the path before a failing edit_file,
    the path is also rolled back."""
    original_content = "<div>original</div>"
    engine, _ = _make_engine(
        initial_content=original_content,
        initial_path="index.html",
    )

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="create_file",
                arguments={"path": "app.html", "content": "<div>new file</div>"},
            ),
            ToolCall(
                id="call-2",
                name="edit_file",
                arguments={"old_text": "NONEXISTENT", "new_text": "fail"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    assert engine.file_state.content == original_content
    assert engine.file_state.path == "index.html"


@pytest.mark.asyncio
async def test_single_failing_edit_no_state_change() -> None:
    """A single failing edit_file does not change file state."""
    original = "<div>content</div>"
    engine, _ = _make_engine(initial_content=original)

    tool_turn = ProviderTurn(
        assistant_text="",
        tool_calls=[
            ToolCall(
                id="call-1",
                name="edit_file",
                arguments={"old_text": "NONEXISTENT", "new_text": "fail"},
            ),
        ],
    )
    final_turn = ProviderTurn(assistant_text="Done", tool_calls=[])
    session = FakeProviderSession([tool_turn, final_turn])

    await engine._run_with_session(session)

    assert engine.file_state.content == original
