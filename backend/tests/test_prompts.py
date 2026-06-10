import pytest
from unittest.mock import patch, MagicMock
import sys
from typing import Any, Dict, List, TypedDict, cast
from openai.types.chat import ChatCompletionMessageParam

# Mock moviepy before importing prompts
sys.modules["moviepy"] = MagicMock()
sys.modules["moviepy.editor"] = MagicMock()

from prompts.pipeline import build_prompt_messages
from prompts.plan import derive_prompt_construction_plan
from prompts.prompt_types import Stack

# Type definitions for test structures
class ExpectedResult(TypedDict):
    messages: List[ChatCompletionMessageParam]


def assert_structure_match(actual: object, expected: object, path: str = "") -> None:
    """
    Compare actual and expected structures with special markers:
    - <ANY>: Matches any value
    - <CONTAINS:text>: Checks if the actual value contains 'text'

    Args:
        actual: The actual value to check
        expected: The expected value or pattern
        path: Current path in the structure (for error messages)
    """
    if (
        isinstance(expected, str)
        and expected.startswith("<")
        and expected.endswith(">")
    ):
        # Handle special markers
        if expected == "<ANY>":
            # Match any value
            return
        elif expected.startswith("<CONTAINS:") and expected.endswith(">"):
            # Extract the text to search for
            search_text = expected[10:-1]  # Remove "<CONTAINS:" and ">"
            assert isinstance(
                actual, str
            ), f"At {path}: expected string, got {type(actual).__name__}"
            assert (
                search_text in actual
            ), f"At {path}: '{search_text}' not found in '{actual}'"
            return

    # Handle different types
    if isinstance(expected, dict):
        assert isinstance(
            actual, dict
        ), f"At {path}: expected dict, got {type(actual).__name__}"
        expected_dict: Dict[str, object] = expected
        actual_dict: Dict[str, object] = actual
        for key, value in expected_dict.items():
            assert key in actual_dict, f"At {path}: key '{key}' not found in actual"
            assert_structure_match(actual_dict[key], value, f"{path}.{key}" if path else key)
    elif isinstance(expected, list):
        assert isinstance(
            actual, list
        ), f"At {path}: expected list, got {type(actual).__name__}"
        expected_list: List[object] = expected
        actual_list: List[object] = actual
        assert len(actual_list) == len(
            expected_list
        ), f"At {path}: list length mismatch (expected {len(expected_list)}, got {len(actual_list)})"
        for i, (a, e) in enumerate(zip(actual_list, expected_list)):
            assert_structure_match(a, e, f"{path}[{i}]")
    else:
        # Direct comparison for other types
        assert actual == expected, f"At {path}: expected {expected}, got {actual}"


class TestCreatePrompt:
    """Test cases for create_prompt function."""

    # Test data constants
    TEST_IMAGE_URL: str = "data:image/png;base64,test_image_data"
    RESULT_IMAGE_URL: str = "data:image/png;base64,result_image_data"
    MOCK_SYSTEM_PROMPT: str = "Mock HTML Tailwind system prompt"
    TEST_STACK: Stack = "html_tailwind"
    ENABLED_IMAGE_POLICY: str = (
        "Image generation is enabled for this request. Use generate_images for "
        "missing assets when needed."
    )

    @staticmethod
    def wrapped_file(content: str) -> str:
        return f'<file path="index.html">\n{content}\n</file>'

    def test_plan_create_uses_create_from_input(self) -> None:
        plan = derive_prompt_construction_plan(
            stack=self.TEST_STACK,
            input_mode="image",
            generation_type="create",
            history=[],
            file_state=None,
        )
        assert plan["construction_strategy"] == "create_from_input"

    @pytest.mark.asyncio
    async def test_create_prompt_includes_design_system(self) -> None:
        messages = await build_prompt_messages(
            stack="html_css",
            input_mode="image",
            generation_type="create",
            prompt={
                "text": "Make a marketing mockup",
                "images": [self.TEST_IMAGE_URL],
                "videos": [],
            },
            history=[],
            image_generation_enabled=True,
            design_system="Reuse .mockup-frame and keep border radius 0.",
        )

        user_content = messages[1].get("content")
        assert isinstance(user_content, list)
        text_part = next(
            part
            for part in user_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
        text = text_part.get("text")
        assert isinstance(text, str)

        assert "## Design system" in text
        assert "Reuse .mockup-frame" in text

    def test_plan_update_with_history_uses_history_strategy(self) -> None:
        plan = derive_prompt_construction_plan(
            stack=self.TEST_STACK,
            input_mode="image",
            generation_type="update",
            history=[{"role": "user", "text": "change", "images": [], "videos": []}],
            file_state=None,
        )
        assert plan["construction_strategy"] == "update_from_history"

    def test_plan_update_without_history_uses_file_snapshot_strategy(self) -> None:
        plan = derive_prompt_construction_plan(
            stack=self.TEST_STACK,
            input_mode="image",
            generation_type="update",
            history=[],
            file_state={"path": "index.html", "content": "<html></html>"},
        )
        assert plan["construction_strategy"] == "update_from_file_snapshot"

    @pytest.mark.asyncio
    async def test_image_mode_create_single_image(self) -> None:
        """Test create generation with single image in image mode."""
        # Setup test data
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "create",
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )

            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {"role": "system", "content": self.MOCK_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.TEST_IMAGE_URL,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": "<CONTAINS:Generate code for a web page that looks exactly like the provided screenshot(s).>",
                            },
                        ],
                    },
                ],
            }

            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_image_mode_create_with_image_generation_disabled(self) -> None:
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "create",
        }

        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=self.MOCK_SYSTEM_PROMPT):
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=[],
                image_generation_enabled=False,
            )

        system_content = messages[0].get("content")
        assert isinstance(system_content, str)
        assert system_content == self.MOCK_SYSTEM_PROMPT

        user_content = messages[1].get("content")
        assert isinstance(user_content, list)
        text_part = next(
            (
                part
                for part in user_content
                if isinstance(part, dict) and part.get("type") == "text"
            ),
            None,
        )
        assert isinstance(text_part, dict)
        user_text = text_part.get("text")
        assert isinstance(user_text, str)
        assert "Image generation is disabled for this request. Do not call generate_images." in user_text


    @pytest.mark.asyncio
    async def test_image_mode_update_with_history(self) -> None:
        """Test update generation with conversation history in image mode."""
        # Setup test data
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "update",
            "history": [
                {"role": "assistant", "text": "<html>Initial code</html>", "images": [], "videos": []},
                {"role": "user", "text": "Make the background blue", "images": [], "videos": []},
                {"role": "assistant", "text": "<html>Updated code</html>", "images": [], "videos": []},
                {"role": "user", "text": "Add a header", "images": [], "videos": []},
            ],
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )

            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Initial code</html>"),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Selected stack: {self.TEST_STACK}.\n\n"
                            f"{self.ENABLED_IMAGE_POLICY}\n\n"
                            "Make the background blue"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Updated code</html>"),
                    },
                    {"role": "user", "content": "Add a header"},
                ],
            }

            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_update_history_with_image_generation_disabled(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=self.MOCK_SYSTEM_PROMPT):
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "", "images": [self.TEST_IMAGE_URL], "videos": []},
                history=[
                    {"role": "assistant", "text": "<html>Initial code</html>", "images": [], "videos": []},
                    {"role": "user", "text": "Make the background blue", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>Updated code</html>", "images": [], "videos": []},
                ],
                image_generation_enabled=False,
            )

        system_content = messages[0].get("content")
        assert isinstance(system_content, str)
        assert system_content == self.MOCK_SYSTEM_PROMPT

        first_user_content = messages[2].get("content")
        assert isinstance(first_user_content, str)
        assert "Selected stack: html_tailwind." in first_user_content
        assert "Image generation is disabled for this request. Do not call generate_images." in first_user_content
        assert "Make the background blue" in first_user_content

    @pytest.mark.asyncio
    async def test_text_mode_create_generation(self) -> None:
        """Test create generation from text description in text mode."""
        # Setup test data
        text_description: str = "a modern landing page with hero section"
        params: Dict[str, Any] = {
            "prompt": {
                "text": text_description,
                "images": []
            },
            "generationType": "create"
        }
        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="text",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )
            
            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"<CONTAINS:Generate UI for {text_description}>"
                    }
                ],
            }
            
            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_text_mode_update_with_history(self) -> None:
        """Test update generation with conversation history in text mode."""
        # Setup test data
        text_description: str = "a dashboard with charts"
        params: Dict[str, Any] = {
            "prompt": {
                "text": text_description,
                "images": []
            },
            "generationType": "update",
            "history": [
                {"role": "assistant", "text": "<html>Initial dashboard</html>", "images": [], "videos": []},
                {"role": "user", "text": "Add a sidebar", "images": [], "videos": []},
                {"role": "assistant", "text": "<html>Dashboard with sidebar</html>", "images": [], "videos": []},
                {"role": "user", "text": "Now add a navigation menu", "images": [], "videos": []},
            ]
        }
        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="text",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )
            
            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Initial dashboard</html>")
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Selected stack: {self.TEST_STACK}.\n\n"
                            f"{self.ENABLED_IMAGE_POLICY}\n\n"
                            "Add a sidebar"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file(
                            "<html>Dashboard with sidebar</html>"
                        )
                    },
                    {
                        "role": "user",
                        "content": "Now add a navigation menu"
                    }
                ],
            }
            
            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_video_mode_basic_prompt_creation(self) -> None:
        """Test basic video prompt creation in video mode.

        For video mode with generation_type="create", we now assemble
        a regular system+user prompt so video generation can run through
        the agent runner path.
        """
        # Setup test data
        video_data_url: str = "data:video/mp4;base64,test_video_data"
        params: Dict[str, Any] = {
            "prompt": {
                "text": "",
                "images": [],
                "videos": [video_data_url],
            },
            "generationType": "create"
        }

        # Call the function
        messages = await build_prompt_messages(
            stack=self.TEST_STACK,
            input_mode="video",
            generation_type=params["generationType"],
            prompt=params["prompt"],
            history=params.get("history", []),
        )

        expected: ExpectedResult = {
            "messages": [
                {
                    "role": "system",
                    "content": "<CONTAINS:You are a coding agent that's an expert at building front-ends.>",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": video_data_url, "detail": "high"},
                        },
                        {
                            "type": "text",
                            "text": "<CONTAINS:Analyze this video and generate the code.>",
                        },
                    ],
                },
            ],
        }

        # Assert the structure matches
        actual: ExpectedResult = {"messages": messages}
        assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_create_raises_on_unsupported_input_mode(self) -> None:
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL], "videos": []},
            "generationType": "create",
        }

        with pytest.raises(ValueError, match="Unsupported input mode: audio"):
            await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode=cast(Any, "audio"),
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=[],
            )


    @pytest.mark.asyncio
    async def test_image_mode_update_with_single_image_in_history(self) -> None:
        """Test update with user message containing a single image."""
        # Setup test data
        reference_image_url: str = "data:image/png;base64,reference_image"
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "update",
            "history": [
                {"role": "assistant", "text": "<html>Initial code</html>", "images": [], "videos": []},
                {"role": "user", "text": "Add a button", "images": [reference_image_url], "videos": []},
                {"role": "assistant", "text": "<html>Code with button</html>", "images": [], "videos": []},
            ]
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )

            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Initial code</html>"),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": reference_image_url,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Selected stack: {self.TEST_STACK}.\n\n"
                                    f"{self.ENABLED_IMAGE_POLICY}\n\n"
                                    "Add a button"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Code with button</html>"),
                    },
                ],
            }

            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_image_mode_update_with_multiple_images_in_history(self) -> None:
        """Test update with user message containing multiple images."""
        # Setup test data
        example1_url: str = "data:image/png;base64,example1"
        example2_url: str = "data:image/png;base64,example2"
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "update",
            "history": [
                {"role": "assistant", "text": "<html>Initial code</html>", "images": [], "videos": []},
                {"role": "user", "text": "Style like these examples", "images": [example1_url, example2_url], "videos": []},
                {"role": "assistant", "text": "<html>Styled code</html>", "images": [], "videos": []},
            ]
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )

            # Define expected structure
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Initial code</html>"),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": example1_url,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": example2_url,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    f"Selected stack: {self.TEST_STACK}.\n\n"
                                    f"{self.ENABLED_IMAGE_POLICY}\n\n"
                                    "Style like these examples"
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Styled code</html>"),
                    },
                ],
            }

            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_update_with_empty_images_arrays(self) -> None:
        """Test that empty images arrays don't break existing functionality."""
        # Setup test data with explicit empty images arrays
        params: Dict[str, Any] = {
            "prompt": {"text": "", "images": [self.TEST_IMAGE_URL]},
            "generationType": "update",
            "history": [
                {"role": "assistant", "text": "<html>Initial code</html>", "images": [], "videos": []},
                {"role": "user", "text": "Make it blue", "images": [], "videos": []},
                {"role": "assistant", "text": "<html>Blue code</html>", "images": [], "videos": []},
            ]
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            # Call the function
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params.get("history", []),
            )

            # Define expected structure - should be text-only messages
            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Initial code</html>"),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Selected stack: {self.TEST_STACK}.\n\n"
                            f"{self.ENABLED_IMAGE_POLICY}\n\n"
                            "Make it blue"
                        ),
                    },  # Text-only message
                    {
                        "role": "assistant",
                        "content": self.wrapped_file("<html>Blue code</html>"),
                    },
                ],
            }

            # Assert the structure matches
            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)

    @pytest.mark.asyncio
    async def test_update_bootstraps_from_file_state_when_history_is_empty(self) -> None:
        """Update should synthesize a user message from fileState + prompt when history is empty."""
        ref_image_url: str = "data:image/png;base64,ref_image"
        params: Dict[str, Any] = {
            "generationType": "update",
            "prompt": {"text": "Make the header blue", "images": [ref_image_url], "videos": []},
            "history": [],
            "fileState": {
                "path": "index.html",
                "content": "<html>Original imported code</html>",
            },
        }

        with patch(
            "prompts.system_prompt.SYSTEM_PROMPT",
            new=self.MOCK_SYSTEM_PROMPT,
        ):
            messages = await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type=params["generationType"],
                prompt=params["prompt"],
                history=params["history"],
                file_state=params["fileState"],
            )

            expected: ExpectedResult = {
                "messages": [
                    {
                        "role": "system",
                        "content": self.MOCK_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": ref_image_url,
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": "<CONTAINS:<current_file path=\"index.html\">>",
                            },
                        ],
                    },
                ],
            }

            actual: ExpectedResult = {"messages": messages}
            assert_structure_match(actual, expected)
            user_content = messages[1].get("content")
            assert isinstance(user_content, list)
            text_part = next(
                (part for part in user_content if isinstance(part, dict) and part.get("type") == "text"),
                None,
            )
            assert isinstance(text_part, dict)
            synthesized_text = text_part.get("text", "")
            assert isinstance(synthesized_text, str)
            assert f"Selected stack: {self.TEST_STACK}." in synthesized_text
            assert "<html>Original imported code</html>" in synthesized_text
            assert "<change_request>" in synthesized_text
            assert "Make the header blue" in synthesized_text

    @pytest.mark.asyncio
    async def test_update_requires_history_or_file_state(self) -> None:
        with pytest.raises(ValueError):
            await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "Change title", "images": [], "videos": []},
                history=[],
            )

    @pytest.mark.asyncio
    async def test_update_history_requires_user_message(self) -> None:
        with pytest.raises(
            ValueError, match="Update history must include at least one user message"
        ):
            await build_prompt_messages(
                stack=self.TEST_STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "Change title", "images": [], "videos": []},
                history=[
                    {
                        "role": "assistant",
                        "text": "<html>Code only</html>",
                        "images": [],
                        "videos": [],
                    }
                ],
            )


# ---------------------------------------------------------------------------
# Unit tests: design_system block builder
# ---------------------------------------------------------------------------
from prompts.design_system import build_design_system_prompt_block


class TestDesignSystemBlock:
    def test_none_returns_empty(self) -> None:
        assert build_design_system_prompt_block(None) == ""

    def test_empty_string_returns_empty(self) -> None:
        assert build_design_system_prompt_block("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert build_design_system_prompt_block("   \n\t  ") == ""

    def test_valid_content_includes_tags(self) -> None:
        result = build_design_system_prompt_block("Use primary color #333.")
        assert "## Design system" in result
        assert "<design_system>" in result
        assert "Use primary color #333." in result
        assert "</design_system>" in result

    def test_content_is_stripped(self) -> None:
        result = build_design_system_prompt_block("  padded content  ")
        assert "<design_system>\npadded content\n</design_system>" in result


# ---------------------------------------------------------------------------
# Unit tests: policies
# ---------------------------------------------------------------------------
from prompts.policies import build_selected_stack_policy, build_user_image_policy


class TestPolicies:
    def test_stack_policy_format(self) -> None:
        assert build_selected_stack_policy("html_tailwind") == "Selected stack: html_tailwind."
        assert build_selected_stack_policy("react_tailwind") == "Selected stack: react_tailwind."

    def test_image_policy_enabled(self) -> None:
        result = build_user_image_policy(True)
        assert "enabled" in result.lower()
        assert "generate_images" in result

    def test_image_policy_disabled(self) -> None:
        result = build_user_image_policy(False)
        assert "disabled" in result.lower()
        assert "Do not call generate_images" in result
        assert "placehold.co" in result


# ---------------------------------------------------------------------------
# Unit tests: message_builder
# ---------------------------------------------------------------------------
from prompts.message_builder import build_history_message, _wrap_assistant_file_content


class TestWrapAssistantFileContent:
    def test_wraps_plain_content(self) -> None:
        result = _wrap_assistant_file_content("<html>hello</html>")
        assert result == '<file path="index.html">\n<html>hello</html>\n</file>'

    def test_does_not_double_wrap(self) -> None:
        already = '<file path="index.html">\n<html>hello</html>\n</file>'
        assert _wrap_assistant_file_content(already) == already

    def test_custom_path(self) -> None:
        result = _wrap_assistant_file_content("<div/>", path="app.html")
        assert result == '<file path="app.html">\n<div/>\n</file>'

    def test_strips_whitespace_before_check(self) -> None:
        padded = '  <file path="x">\ncontent\n</file>  '
        result = _wrap_assistant_file_content(padded)
        assert result.startswith('<file path="x">')


class TestBuildHistoryMessage:
    @staticmethod
    def _build(item: Any) -> dict[str, Any]:
        return cast(dict[str, Any], build_history_message(item))

    def test_user_text_only(self) -> None:
        msg = self._build(
            {"role": "user", "text": "hello", "images": [], "videos": []}
        )
        assert msg["role"] == "user"
        assert msg["content"] == "hello"

    def test_user_with_images(self) -> None:
        msg = self._build(
            {"role": "user", "text": "look", "images": ["img1", "img2"], "videos": []}
        )
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 3  # 2 images + 1 text
        assert content[0]["type"] == "image_url"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "text"

    def test_user_with_video(self) -> None:
        msg = self._build(
            {"role": "user", "text": "watch", "images": [], "videos": ["vid1"]}
        )
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 2  # 1 video + 1 text
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "vid1"

    def test_user_with_images_and_videos(self) -> None:
        msg = self._build(
            {"role": "user", "text": "both", "images": ["img1"], "videos": ["vid1"]}
        )
        content = msg["content"]
        assert isinstance(content, list)
        assert len(content) == 3  # img1 + vid1 + text

    def test_assistant_wraps_content(self) -> None:
        msg = self._build(
            {"role": "assistant", "text": "<html>code</html>", "images": [], "videos": []}
        )
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert isinstance(content, str)
        assert '<file path="index.html">' in content
        assert "<html>code</html>" in content

    def test_assistant_already_wrapped(self) -> None:
        wrapped = '<file path="index.html">\n<html>code</html>\n</file>'
        msg = self._build(
            {"role": "assistant", "text": wrapped, "images": [], "videos": []}
        )
        assert msg["content"] == wrapped


# ---------------------------------------------------------------------------
# Unit tests: prompt construction plan edge cases
# ---------------------------------------------------------------------------


class TestPromptConstructionPlanEdgeCases:
    STACK: Stack = "html_tailwind"

    def test_create_always_returns_create_from_input(self) -> None:
        for mode in ("image", "text", "video"):
            plan = derive_prompt_construction_plan(
                stack=self.STACK,
                input_mode=mode,  # type: ignore[arg-type]
                generation_type="create",
                history=[],
                file_state=None,
            )
            assert plan["construction_strategy"] == "create_from_input"
            assert plan["input_mode"] == mode

    def test_update_empty_history_none_file_state_raises(self) -> None:
        with pytest.raises(ValueError, match="Update requests require"):
            derive_prompt_construction_plan(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                history=[],
                file_state=None,
            )

    def test_update_empty_history_blank_file_content_raises(self) -> None:
        with pytest.raises(ValueError, match="Update requests require"):
            derive_prompt_construction_plan(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                history=[],
                file_state={"path": "index.html", "content": "   "},
            )

    def test_update_empty_history_empty_file_content_raises(self) -> None:
        with pytest.raises(ValueError, match="Update requests require"):
            derive_prompt_construction_plan(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                history=[],
                file_state={"path": "index.html", "content": ""},
            )

    def test_update_history_takes_priority_over_file_state(self) -> None:
        plan = derive_prompt_construction_plan(
            stack=self.STACK,
            input_mode="image",
            generation_type="update",
            history=[{"role": "user", "text": "change", "images": [], "videos": []}],
            file_state={"path": "index.html", "content": "<html/>"},
        )
        assert plan["construction_strategy"] == "update_from_history"

    def test_plan_preserves_stack(self) -> None:
        for stack in ("html_css", "react_tailwind", "bootstrap"):
            plan = derive_prompt_construction_plan(
                stack=stack,  # type: ignore[arg-type]
                input_mode="text",
                generation_type="create",
                history=[],
                file_state=None,
            )
            assert plan["stack"] == stack


# ---------------------------------------------------------------------------
# Integration tests: update_from_file_snapshot edge cases
# ---------------------------------------------------------------------------
MOCK_SYS = "MOCK_SYSTEM"

# Messages are plain dicts at runtime, but typed as a ChatCompletionMessageParam
# union whose fields are not all required.  This helper avoids verbose per-access
# casts in test assertions.
def _p(prompt: Any) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], prompt)


def _parts(msgs: list[dict[str, Any]], index: int) -> list[dict[str, Any]]:
    """Extract multipart content from message *index*, asserting it is a list."""
    c = msgs[index]["content"]
    assert isinstance(c, list)
    return cast(list[dict[str, Any]], c)


class TestUpdateFromFileSnapshot:
    STACK: Stack = "html_tailwind"
    FILE_STATE: Dict[str, str] = {
        "path": "index.html",
        "content": "<html><body>old</body></html>",
    }

    @pytest.mark.asyncio
    async def test_empty_text_uses_default_change_request(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        user_text = self._extract_user_text(messages)
        assert "Apply the requested update." in user_text

    @pytest.mark.asyncio
    async def test_whitespace_only_text_uses_default(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "   \n  ", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        user_text = self._extract_user_text(messages)
        assert "Apply the requested update." in user_text

    @pytest.mark.asyncio
    async def test_pure_text_no_images(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Add a footer", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        m = _p(messages)
        assert len(m) == 2
        # No images → content should be a plain string, not a list
        user_content = m[1]["content"]
        assert isinstance(user_content, str)
        assert "Add a footer" in user_content
        assert "<current_file" in user_content

    @pytest.mark.asyncio
    async def test_image_generation_disabled(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Fix styling", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
                image_generation_enabled=False,
            )
        user_text = self._extract_user_text(messages)
        assert "Do not call generate_images" in user_text

    @pytest.mark.asyncio
    async def test_with_design_system(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Change color", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
                design_system="Use brand color #FF0000.",
            )
        user_text = self._extract_user_text(messages)
        assert "## Design system" in user_text
        assert "Use brand color #FF0000." in user_text

    @pytest.mark.asyncio
    async def test_empty_design_system_omitted(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Change color", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
                design_system="",
            )
        user_text = self._extract_user_text(messages)
        assert "## Design system" not in user_text

    @pytest.mark.asyncio
    async def test_whitespace_design_system_omitted(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Change color", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
                design_system="   \n  ",
            )
        user_text = self._extract_user_text(messages)
        assert "## Design system" not in user_text

    @pytest.mark.asyncio
    async def test_custom_file_path(self) -> None:
        custom_state = {"path": "components/App.vue", "content": "<template/>"}
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack="vue_tailwind",
                input_mode="text",
                generation_type="update",
                prompt={"text": "Add prop", "images": [], "videos": []},
                history=[],
                file_state=custom_state,
            )
        user_text = self._extract_user_text(messages)
        assert '<current_file path="components/App.vue">' in user_text

    @pytest.mark.asyncio
    async def test_missing_path_key_defaults_to_index_html(self) -> None:
        state_no_path: Dict[str, str] = {"content": "<html>no path</html>"}
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Update", "images": [], "videos": []},
                history=[],
                file_state=state_no_path,
            )
        user_text = self._extract_user_text(messages)
        assert '<current_file path="index.html">' in user_text

    @pytest.mark.asyncio
    async def test_file_content_embedded_in_prompt(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "Refactor", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        user_text = self._extract_user_text(messages)
        assert "<html><body>old</body></html>" in user_text
        assert "<change_request>" in user_text
        assert "Refactor" in user_text

    @pytest.mark.asyncio
    async def test_with_reference_image(self) -> None:
        ref_img = "data:image/png;base64,snapshot_ref"
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "Match this design", "images": [ref_img], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        m = _p(messages)
        parts = _parts(m, 1)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"] == ref_img

    @pytest.mark.asyncio
    async def test_stack_policy_present(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack="react_tailwind",
                input_mode="text",
                generation_type="update",
                prompt={"text": "Add hook", "images": [], "videos": []},
                history=[],
                file_state=self.FILE_STATE,
            )
        user_text = self._extract_user_text(messages)
        assert "Selected stack: react_tailwind." in user_text

    @staticmethod
    def _extract_user_text(messages: List[Any]) -> str:
        m = cast(list[dict[str, Any]], messages)
        content = m[1]["content"]
        if isinstance(content, str):
            return content
        parts = cast(list[dict[str, Any]], content)
        part = next(p for p in parts if p.get("type") == "text")
        result: str = part["text"]
        return result


# ---------------------------------------------------------------------------
# Integration tests: update_from_history edge cases
# ---------------------------------------------------------------------------


class TestUpdateFromHistoryEdgeCases:
    STACK: Stack = "html_tailwind"

    @pytest.mark.asyncio
    async def test_with_design_system(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "Build a form", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>form</html>", "images": [], "videos": []},
                ],
                design_system="Use .btn-primary for all buttons.",
            )
        m = _p(messages)
        first_user = m[1]["content"]
        assert isinstance(first_user, str)
        assert "## Design system" in first_user
        assert "Use .btn-primary for all buttons." in first_user
        assert "Build a form" in first_user

    @pytest.mark.asyncio
    async def test_none_design_system_omitted(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "Hello", "images": [], "videos": []},
                ],
                design_system=None,
            )
        m = _p(messages)
        first_user = m[1]["content"]
        assert isinstance(first_user, str)
        assert "## Design system" not in first_user

    @pytest.mark.asyncio
    async def test_minimal_history_single_user_message(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "Create a page", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        assert len(m) == 2  # system + 1 user
        assert m[0]["role"] == "system"
        assert m[1]["role"] == "user"
        user_text = m[1]["content"]
        assert isinstance(user_text, str)
        assert "Selected stack:" in user_text
        assert "Create a page" in user_text

    @pytest.mark.asyncio
    async def test_first_user_empty_text(self) -> None:
        """First user message with empty text should still get stack prefix."""
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "", "images": ["img1"], "videos": []},
                    {"role": "assistant", "text": "<html>v1</html>", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        parts = _parts(m, 1)
        text_part: dict[str, Any] = next(
            p for p in parts if p.get("type") == "text"
        )
        # Stack prefix present even with empty user text
        assert "Selected stack:" in text_part["text"]

    @pytest.mark.asyncio
    async def test_first_user_whitespace_only_text(self) -> None:
        """Whitespace-only text treated same as empty (stack prefix only)."""
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "   ", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>v1</html>", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        first_user = m[1]["content"]
        assert isinstance(first_user, str)
        assert "Selected stack:" in first_user
        # No trailing user text beyond the prefix
        assert first_user.endswith(".")

    @pytest.mark.asyncio
    async def test_history_with_video_in_user_message(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="video",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "Build from video", "images": [], "videos": ["vid_url"]},
                    {"role": "assistant", "text": "<html>from video</html>", "images": [], "videos": []},
                    {"role": "user", "text": "Add animation", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        # First user message should have video as image_url content part
        parts = _parts(m, 1)
        media_parts: list[dict[str, Any]] = [
            p for p in parts
            if p.get("type") == "image_url"
        ]
        assert len(media_parts) == 1
        assert media_parts[0]["image_url"]["url"] == "vid_url"
        # Subsequent user message is plain text
        last_user = m[3]
        assert last_user["role"] == "user"
        assert last_user["content"] == "Add animation"

    @pytest.mark.asyncio
    async def test_subsequent_user_messages_no_prefix(self) -> None:
        """Only the first user message gets the stack/policy prefix."""
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "First", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>v1</html>", "images": [], "videos": []},
                    {"role": "user", "text": "Second", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>v2</html>", "images": [], "videos": []},
                    {"role": "user", "text": "Third", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        # First user gets prefix
        assert "Selected stack:" in str(m[1]["content"])
        # Third and fifth messages (user) have no prefix
        assert m[3]["content"] == "Second"
        assert m[5]["content"] == "Third"

    @pytest.mark.asyncio
    async def test_assistant_messages_wrapped_in_file_tag(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "user", "text": "Start", "images": [], "videos": []},
                    {"role": "assistant", "text": "<html>code</html>", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        assistant_content = m[2]["content"]
        assert isinstance(assistant_content, str)
        assert '<file path="index.html">' in assistant_content

    @pytest.mark.asyncio
    async def test_history_starts_with_assistant(self) -> None:
        """When history starts with assistant, first user still gets prefix."""
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="update",
                prompt={"text": "", "images": [], "videos": []},
                history=[
                    {"role": "assistant", "text": "<html>initial</html>", "images": [], "videos": []},
                    {"role": "user", "text": "Update it", "images": [], "videos": []},
                ],
            )
        m = _p(messages)
        # system + assistant + user = 3
        assert len(m) == 3
        assert m[1]["role"] == "assistant"
        user_msg = m[2]["content"]
        assert isinstance(user_msg, str)
        assert "Selected stack:" in user_msg
        assert "Update it" in user_msg


# ---------------------------------------------------------------------------
# Integration tests: create prompt edge cases
# ---------------------------------------------------------------------------


class TestCreateEdgeCases:
    STACK: Stack = "html_tailwind"

    @pytest.mark.asyncio
    async def test_text_mode_empty_prompt(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="create",
                prompt={"text": "", "images": [], "videos": []},
                history=[],
            )
        m = _p(messages)
        assert len(m) == 2
        user_text = m[1]["content"]
        assert isinstance(user_text, str)
        assert "Generate UI for " in user_text

    @pytest.mark.asyncio
    async def test_text_mode_image_generation_disabled(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="create",
                prompt={"text": "dashboard", "images": [], "videos": []},
                history=[],
                image_generation_enabled=False,
            )
        m = _p(messages)
        user_text = m[1]["content"]
        assert isinstance(user_text, str)
        assert "Do not call generate_images" in user_text

    @pytest.mark.asyncio
    async def test_text_mode_with_design_system(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="text",
                generation_type="create",
                prompt={"text": "landing page", "images": [], "videos": []},
                history=[],
                design_system="Font: Inter. Radius: 8px.",
            )
        m = _p(messages)
        user_text = m[1]["content"]
        assert isinstance(user_text, str)
        assert "## Design system" in user_text
        assert "Font: Inter. Radius: 8px." in user_text

    @pytest.mark.asyncio
    async def test_image_mode_multiple_images(self) -> None:
        imgs = ["data:image/png;base64,img1", "data:image/png;base64,img2", "data:image/png;base64,img3"]
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="create",
                prompt={"text": "", "images": imgs, "videos": []},
                history=[],
            )
        parts = _parts(_p(messages), 1)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 3

    @pytest.mark.asyncio
    async def test_image_mode_with_additional_text(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="create",
                prompt={"text": "Make it dark theme", "images": ["img_url"], "videos": []},
                history=[],
            )
        parts = _parts(_p(messages), 1)
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "Additional instructions: Make it dark theme" in text_part["text"]

    @pytest.mark.asyncio
    async def test_image_mode_no_additional_text(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="create",
                prompt={"text": "", "images": ["img_url"], "videos": []},
                history=[],
            )
        parts = _parts(_p(messages), 1)
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "Additional instructions:" not in text_part["text"]

    @pytest.mark.asyncio
    async def test_video_mode_no_video_raises(self) -> None:
        with pytest.raises(ValueError, match="Video mode requires a video"):
            await build_prompt_messages(
                stack=self.STACK,
                input_mode="video",
                generation_type="create",
                prompt={"text": "Build it", "images": [], "videos": []},
                history=[],
            )

    @pytest.mark.asyncio
    async def test_video_mode_image_generation_disabled(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="video",
                generation_type="create",
                prompt={"text": "", "images": [], "videos": ["vid_url"]},
                history=[],
                image_generation_enabled=False,
            )
        parts = _parts(_p(messages), 1)
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "Do not call generate_images" in text_part["text"]

    @pytest.mark.asyncio
    async def test_video_mode_with_design_system(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="video",
                generation_type="create",
                prompt={"text": "", "images": [], "videos": ["vid_url"]},
                history=[],
                design_system="Brand guide v2.",
            )
        parts = _parts(_p(messages), 1)
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "Brand guide v2." in text_part["text"]

    @pytest.mark.asyncio
    async def test_video_mode_with_additional_text(self) -> None:
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="video",
                generation_type="create",
                prompt={"text": "Focus on the nav bar", "images": [], "videos": ["vid_url"]},
                history=[],
            )
        parts = _parts(_p(messages), 1)
        text_part = next(p for p in parts if p.get("type") == "text")
        assert "Additional instructions: Focus on the nav bar" in text_part["text"]

    @pytest.mark.asyncio
    async def test_image_mode_empty_images_list(self) -> None:
        """Image mode with empty images list still builds without error."""
        with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
            messages = await build_prompt_messages(
                stack=self.STACK,
                input_mode="image",
                generation_type="create",
                prompt={"text": "Build a page", "images": [], "videos": []},
                history=[],
            )
        m = _p(messages)
        assert len(m) == 2
        # With no images the user content list has only the text part
        parts = _parts(m, 1)
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 0

    @pytest.mark.asyncio
    async def test_different_stacks_in_create(self) -> None:
        """Each stack name appears in the generated prompt."""
        for stack in ("html_css", "react_tailwind", "bootstrap", "ionic_tailwind", "vue_tailwind"):
            with patch("prompts.system_prompt.SYSTEM_PROMPT", new=MOCK_SYS):
                messages = await build_prompt_messages(
                    stack=stack,  # type: ignore[arg-type]
                    input_mode="text",
                    generation_type="create",
                    prompt={"text": "a page", "images": [], "videos": []},
                    history=[],
                )
            m = _p(messages)
            user_text = m[1]["content"]
            assert isinstance(user_text, str)
            assert f"Selected stack: {stack}." in user_text
