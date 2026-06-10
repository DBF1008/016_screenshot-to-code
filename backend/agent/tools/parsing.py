# pyright: reportUnknownVariableType=false
import json
from typing import Any, Dict, Optional, Tuple

from agent.state import ensure_str


def parse_json_arguments(raw_args: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    if isinstance(raw_args, dict):
        return raw_args, None
    if raw_args is None:
        return {}, None
    raw_text = ensure_str(raw_args).strip()
    if not raw_text:
        return {}, None
    try:
        return json.loads(raw_text), None
    except json.JSONDecodeError as exc:
        return {}, f"Invalid JSON arguments: {exc}"


def _strip_incomplete_escape(value: str) -> str:
    if not value:
        return value
    trailing = 0
    for ch in reversed(value):
        if ch == "\\":
            trailing += 1
        else:
            break
    if trailing % 2 == 1:
        return value[:-1]
    return value


def _extract_partial_json_string(raw_text: str, key: str) -> Optional[str]:
    if not raw_text:
        return None
    token = f'"{key}"'
    idx = raw_text.find(token)
    if idx == -1:
        return None
    colon = raw_text.find(":", idx + len(token))
    if colon == -1:
        return None
    cursor = colon + 1
    while cursor < len(raw_text) and raw_text[cursor].isspace():
        cursor += 1
    if cursor >= len(raw_text) or raw_text[cursor] != '"':
        return None

    start = cursor + 1
    last_quote: Optional[int] = None
    cursor = start
    while cursor < len(raw_text):
        if raw_text[cursor] == '"':
            backslashes = 0
            back = cursor - 1
            while back >= start and raw_text[back] == "\\":
                backslashes += 1
                back -= 1
            if backslashes % 2 == 0:
                last_quote = cursor
        cursor += 1

    partial = raw_text[start:] if last_quote is None else raw_text[start:last_quote]
    partial = _strip_incomplete_escape(partial)
    if not partial:
        return ""

    try:
        return json.loads(f'"{partial}"')
    except Exception:
        return (
            partial.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\\\", "\\")
        )


def extract_content_from_args(raw_args: Any) -> Optional[str]:
    if isinstance(raw_args, dict):
        content = raw_args.get("content")
        if content is None:
            return None
        return ensure_str(content)
    raw_text = ensure_str(raw_args)
    return _extract_partial_json_string(raw_text, "content")


def extract_path_from_args(raw_args: Any) -> Optional[str]:
    if isinstance(raw_args, dict):
        path = raw_args.get("path")
        return ensure_str(path) if path is not None else None
    raw_text = ensure_str(raw_args)
    return _extract_partial_json_string(raw_text, "path")


def _extract_partial_json_string_with_status(
    raw_text: str, key: str
) -> Tuple[Optional[str], bool]:
    """Like _extract_partial_json_string but also returns whether the value is complete."""
    if not raw_text:
        return None, False
    token = f'"{key}"'
    idx = raw_text.find(token)
    if idx == -1:
        return None, False
    colon = raw_text.find(":", idx + len(token))
    if colon == -1:
        return None, False
    cursor = colon + 1
    while cursor < len(raw_text) and raw_text[cursor].isspace():
        cursor += 1
    if cursor >= len(raw_text) or raw_text[cursor] != '"':
        return None, False

    start = cursor + 1
    last_quote: Optional[int] = None
    cursor = start
    while cursor < len(raw_text):
        if raw_text[cursor] == '"':
            backslashes = 0
            back = cursor - 1
            while back >= start and raw_text[back] == "\\":
                backslashes += 1
                back -= 1
            if backslashes % 2 == 0:
                last_quote = cursor
        cursor += 1

    is_complete = last_quote is not None
    partial = raw_text[start:] if last_quote is None else raw_text[start:last_quote]
    partial = _strip_incomplete_escape(partial)
    if not partial:
        return "", is_complete

    try:
        return json.loads(f'"{partial}"'), is_complete
    except Exception:
        return (
            partial.replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace('\\"', '"')
            .replace("\\\\", "\\"),
            is_complete,
        )


def _find_complete_objects_in_array(
    raw_text: str, array_key: str
) -> list[Dict[str, Any]]:
    """Find complete JSON objects inside a (possibly incomplete) JSON array."""
    token = f'"{array_key}"'
    idx = raw_text.find(token)
    if idx == -1:
        return []

    bracket = raw_text.find("[", idx + len(token))
    if bracket == -1:
        return []

    results: list[Dict[str, Any]] = []
    cursor = bracket + 1
    depth = 0
    in_string = False
    obj_start = -1

    while cursor < len(raw_text):
        ch = raw_text[cursor]

        if in_string:
            if ch == "\\":
                cursor += 2
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    obj_start = cursor
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and obj_start >= 0:
                    obj_text = raw_text[obj_start : cursor + 1]
                    try:
                        obj = json.loads(obj_text)
                        if isinstance(obj, dict):
                            results.append(obj)
                    except json.JSONDecodeError:
                        pass
                    obj_start = -1
            elif ch == "]" and depth == 0:
                break

        cursor += 1

    return results


def extract_completed_edits_from_args(raw_args: Any) -> list[Dict[str, Any]]:
    """Extract completed edit pairs from partial or complete edit_file arguments.

    Returns a list of dicts with ``old_text``, ``new_text``, and optionally ``count``.
    Only edits where both text fields are fully streamed are included.
    """
    if isinstance(raw_args, dict):
        edits = raw_args.get("edits")
        if isinstance(edits, list):
            return [
                e
                for e in edits
                if isinstance(e, dict) and "old_text" in e and "new_text" in e
            ]
        old = raw_args.get("old_text")
        new = raw_args.get("new_text")
        if old is not None and new is not None:
            result: Dict[str, Any] = {"old_text": old, "new_text": new}
            count = raw_args.get("count")
            if count is not None:
                result["count"] = count
            return [result]
        return []

    raw_text = ensure_str(raw_args)
    if not raw_text:
        return []

    # Batch format
    if '"edits"' in raw_text:
        objects = _find_complete_objects_in_array(raw_text, "edits")
        return [obj for obj in objects if "old_text" in obj and "new_text" in obj]

    # Single edit format — only return when both values are complete
    old_text, old_complete = _extract_partial_json_string_with_status(
        raw_text, "old_text"
    )
    if old_text is None or not old_complete:
        return []
    new_text, new_complete = _extract_partial_json_string_with_status(
        raw_text, "new_text"
    )
    if new_text is None or not new_complete:
        return []

    return [{"old_text": old_text, "new_text": new_text}]
