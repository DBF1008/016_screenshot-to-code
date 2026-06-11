import base64
from pathlib import Path

import pytest

from agent.state import AgentFileState
from agent.tools.runtime import AgentToolRuntime
from agent.tools.types import ToolCall
from uploaded_assets import persist_data_url_as_temporary_asset


def _data_url(payload: bytes, content_type: str = "image/png") -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def test_edit_file_returns_structured_result_with_diff() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<div>before</div>\n<p>keep</p>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = runtime._edit_file(
        {
            "old_text": "<div>before</div>",
            "new_text": "<div>after</div>",
        }
    )

    assert result.ok is True
    assert result.updated_content == "<div>after</div>\n<p>keep</p>\n"
    assert result.result["content"] == "Successfully edited file at index.html."
    assert set(result.result["details"].keys()) == {"diff", "firstChangedLine"}
    assert result.result["details"]["firstChangedLine"] == 1
    assert "--- index.html" in result.result["details"]["diff"]
    assert "+++ index.html" in result.result["details"]["diff"]
    assert "-<div>before</div>" in result.result["details"]["diff"]
    assert "+<div>after</div>" in result.result["details"]["diff"]
    assert result.summary["firstChangedLine"] == 1
    assert result.summary["diff"] == result.result["details"]["diff"]


@pytest.mark.asyncio
async def test_execute_edit_file_uses_updated_result_shape() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(path="index.html", content="<main>old</main>"),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = await runtime.execute(
        ToolCall(
            id="call-1",
            name="edit_file",
            arguments={"old_text": "old", "new_text": "new"},
        )
    )

    # execute() is sync for edit_file and should preserve the structured payload.
    assert result.ok is True
    assert result.result["content"] == "Successfully edited file at index.html."
    assert set(result.result["details"].keys()) == {"diff", "firstChangedLine"}
    assert "--- index.html" in result.result["details"]["diff"]


@pytest.mark.asyncio
async def test_save_assets_promotes_temporary_asset_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temp_dir = tmp_path / "tmp-assets"
    asset_dir = tmp_path / "local-assets"
    monkeypatch.setattr("uploaded_assets.store.TEMP_ASSET_DIR", str(temp_dir))
    monkeypatch.setattr("uploaded_assets.store.LOCAL_ASSET_DIR", str(asset_dir))

    temp_asset = persist_data_url_as_temporary_asset(
        _data_url(b"image-bytes"),
        "http://127.0.0.1:7001",
    )
    assert temp_asset is not None

    runtime = AgentToolRuntime(
        file_state=AgentFileState(),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = await runtime.execute(
        ToolCall(
            id="call-1",
            name="save_assets",
            arguments={"asset_ids": [temp_asset.asset_id]},
        )
    )

    assert result.ok is True
    images = result.result["images"]
    assert len(images) == 1
    assert images[0]["asset_id"] == temp_asset.asset_id
    assert images[0]["status"] == "ok"
    assert images[0]["public_url"].startswith(
        "http://127.0.0.1:7001/local-assets/"
    )
    assert temp_asset.asset_id not in images[0]["public_url"]
    assert len(list(asset_dir.iterdir())) == 1


def test_batch_edit_all_succeed() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<h1>old title</h1>\n<p>old body</p>\n<footer>old footer</footer>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = runtime._edit_file(
        {
            "edits": [
                {"old_text": "<h1>old title</h1>", "new_text": "<h1>new title</h1>"},
                {"old_text": "<p>old body</p>", "new_text": "<p>new body</p>"},
                {"old_text": "<footer>old footer</footer>", "new_text": "<footer>new footer</footer>"},
            ]
        }
    )

    assert result.ok is True
    assert result.updated_content == (
        "<h1>new title</h1>\n<p>new body</p>\n<footer>new footer</footer>\n"
    )
    assert runtime.file_state.content == result.updated_content
    assert result.result["content"] == "Successfully edited file at index.html."
    assert "failed" not in result.result


def test_batch_edit_partial_failure_commits_successful_edits() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<h1>title</h1>\n<p>body</p>\n<footer>footer</footer>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = runtime._edit_file(
        {
            "edits": [
                {"old_text": "<h1>title</h1>", "new_text": "<h1>new title</h1>"},
                {"old_text": "<p>nonexistent</p>", "new_text": "<p>replacement</p>"},
                {"old_text": "<footer>footer</footer>", "new_text": "<footer>new footer</footer>"},
            ]
        }
    )

    # ok=False because one edit failed
    assert result.ok is False

    # But the two successful edits are committed
    assert "<h1>new title</h1>" in runtime.file_state.content
    assert "<footer>new footer</footer>" in runtime.file_state.content
    # The failed edit's old_text was not present, so body is unchanged
    assert "<p>body</p>" in runtime.file_state.content

    # updated_content reflects the partial changes
    assert result.updated_content == runtime.file_state.content

    # Detailed per-edit breakdown
    assert len(result.result["succeeded"]) == 2
    assert result.result["succeeded"][0]["index"] == 0
    assert result.result["succeeded"][1]["index"] == 2
    assert len(result.result["failed"]) == 1
    assert result.result["failed"][0]["index"] == 1
    assert result.result["failed"][0]["error"] == "old_text not found"

    # Summary includes the failure breakdown
    assert "1 of 3 edit(s) failed" in result.result["error"]
    assert len(result.summary["failed"]) == 1


def test_batch_edit_all_fail() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<h1>title</h1>\n<p>body</p>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    original_content = runtime.file_state.content
    result = runtime._edit_file(
        {
            "edits": [
                {"old_text": "<div>nope</div>", "new_text": "<div>x</div>"},
                {"old_text": "<span>nada</span>", "new_text": "<span>y</span>"},
            ]
        }
    )

    assert result.ok is False
    # Content unchanged when all edits fail
    assert runtime.file_state.content == original_content
    assert result.updated_content is None
    assert len(result.result["failed"]) == 2
    assert result.result["succeeded"] == []


def test_single_edit_failure_unchanged_behavior() -> None:
    """Single (non-batch) edit that fails should behave the same as before."""
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<h1>title</h1>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    original_content = runtime.file_state.content
    result = runtime._edit_file(
        {"old_text": "<div>nonexistent</div>", "new_text": "<div>replacement</div>"}
    )

    assert result.ok is False
    assert runtime.file_state.content == original_content
    assert result.updated_content is None


def test_batch_edit_diff_reflects_actual_changes() -> None:
    runtime = AgentToolRuntime(
        file_state=AgentFileState(
            path="index.html",
            content="<h1>old</h1>\n<p>keep</p>\n<footer>old</footer>\n",
        ),
        should_generate_images=False,
        openai_api_key=None,
        openai_base_url=None,
    )

    result = runtime._edit_file(
        {
            "edits": [
                {"old_text": "<h1>old</h1>", "new_text": "<h1>new</h1>"},
                {"old_text": "<span>missing</span>", "new_text": "<span>x</span>"},
            ]
        }
    )

    assert result.ok is False
    # Diff should show the successful h1 change
    assert "-<h1>old</h1>" in result.summary["diff"]
    assert "+<h1>new</h1>" in result.summary["diff"]
    # The failed edit should not appear in the diff
    assert "missing" not in result.summary["diff"]
    assert "<span>" not in result.summary["diff"]
