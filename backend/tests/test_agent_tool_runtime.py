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
    diff = result.result["details"]["diff"]
    assert "index.html" in diff
    assert "<div>before</div>" in diff
    assert "<div>after</div>" in diff
    assert "->" in diff
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
    assert "->" in result.result["details"]["diff"]


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


def test_generate_compact_diff_single_line_change() -> None:
    result = AgentToolRuntime._generate_diff(
        "<p>old text</p>\n<div>keep</div>\n",
        "<p>new text</p>\n<div>keep</div>\n",
        "index.html",
    )
    assert "->" in result["diff"]
    assert "old text" in result["diff"]
    assert "new text" in result["diff"]
    assert result["firstChangedLine"] == 1
    # Should NOT contain unified diff markers
    assert "---" not in result["diff"]
    assert "+++" not in result["diff"]


def test_generate_diff_falls_back_to_unified_for_large_changes() -> None:
    old = "\n".join(f"<p>line {i}</p>" for i in range(20))
    new = "\n".join(f"<p>changed {i}</p>" for i in range(20))
    result = AgentToolRuntime._generate_diff(old, new, "index.html")
    assert "---" in result["diff"]
    assert "+++" in result["diff"]


def test_generate_compact_diff_line_deletion() -> None:
    result = AgentToolRuntime._generate_diff(
        "<h1>Title</h1>\n<p>remove me</p>\n<footer>End</footer>\n",
        "<h1>Title</h1>\n<footer>End</footer>\n",
        "index.html",
    )
    assert 'deleted' in result["diff"]
    assert "remove me" in result["diff"]
    assert result["firstChangedLine"] is not None


def test_generate_compact_diff_line_insertion() -> None:
    result = AgentToolRuntime._generate_diff(
        "<h1>Title</h1>\n<footer>End</footer>\n",
        "<h1>Title</h1>\n<p>new line</p>\n<footer>End</footer>\n",
        "index.html",
    )
    assert 'inserted' in result["diff"]
    assert "new line" in result["diff"]
    assert result["firstChangedLine"] is not None
