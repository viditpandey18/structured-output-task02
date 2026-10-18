"""Automated test suite for Task 02 — Structured Output.

Covers all required acceptance criteria:
1. Valid JSON returns a TaskResult object.
2. Invalid output is successfully repaired.
3. Invalid output is rejected when all attempts fail.
4. Unexpected extra fields are rejected.
5. Empty tags are rejected.
6. Malformed JSON is handled safely.
7. The repair callback is not called after the final allowed attempt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    TaskResult,
    StructuredOutputError,
    validate_with_retry,
    mock_repair_callback,
)


# ============================================================================
# Fixtures & Sample Payloads
# ============================================================================

@pytest.fixture
def valid_payload_dict() -> dict:
    """Returns a valid dictionary payload adhering strictly to TaskResult schema."""
    return {
        "title": "Implement OAuth2 token refresh logic",
        "summary": "Handle expired JWT access tokens by silently issuing refresh token requests.",
        "priority": "high",
        "estimated_hours": 12.5,
        "tags": ["auth", "security", "api"],
    }


@pytest.fixture
def valid_json_str(valid_payload_dict: dict) -> str:
    """Returns valid JSON text representing a TaskResult."""
    return json.dumps(valid_payload_dict)


# ============================================================================
# Requirement 1: Valid JSON returns a TaskResult object
# ============================================================================

def test_valid_json_returns_task_result(valid_json_str: str) -> None:
    """Verify that syntactically valid JSON matching schema returns a TaskResult instance."""
    result = validate_with_retry(valid_json_str)

    assert isinstance(result, TaskResult)
    assert result.title == "Implement OAuth2 token refresh logic"
    assert result.priority == "high"
    assert result.estimated_hours == 12.5
    assert result.tags == ["auth", "security", "api"]
    assert len(result.summary) >= 10


# ============================================================================
# Requirement 2: Invalid output is successfully repaired
# ============================================================================

def test_invalid_output_is_successfully_repaired() -> None:
    """Verify that an output with multiple validation issues is repaired via callback."""
    flawed_output = json.dumps({
        "title": "Fix memory leak in background worker",
        "summary": "Investigate Celery worker memory consumption under peak loads.",
        "priority": "critical",  # Invalid priority (schema allows: low, medium, high)
        "estimated_hours": 350,   # Exceeds max 200
        "tags": [],               # Empty tags list
        "disallowed_key": "extra", # Extra field forbidden
    })

    result = validate_with_retry(
        raw_output=flawed_output,
        repair_callback=mock_repair_callback,
        max_attempts=3,
    )

    assert isinstance(result, TaskResult)
    assert result.title == "Fix memory leak in background worker"
    assert result.priority == "high"  # Repaired from critical
    assert result.estimated_hours == 40.0  # Clamped to valid range
    assert len(result.tags) >= 1  # Repaired with default valid tags
    assert not hasattr(result, "disallowed_key")


# ============================================================================
# Requirement 3: Invalid output is rejected when all attempts fail
# ============================================================================

def test_invalid_output_rejected_when_all_attempts_fail() -> None:
    """Verify that StructuredOutputError is raised when repair callback fails to fix issues."""
    always_invalid_json = json.dumps({"title": "X"})  # Missing required fields & title too short

    def failing_repair_callback(output: str, err: Exception) -> str:
        # Returns still-invalid JSON on every repair call
        return json.dumps({"title": "Still too short", "tags": []})

    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(
            raw_output=always_invalid_json,
            repair_callback=failing_repair_callback,
            max_attempts=3,
        )

    assert "Validation failed after 3 attempt(s)" in str(exc_info.value)
    assert len(exc_info.value.errors) == 3


def test_invalid_output_rejected_without_repair_callback(valid_payload_dict: dict) -> None:
    """Verify that when no repair callback is provided, failed validation immediately errors."""
    valid_payload_dict["estimated_hours"] = -10  # Must be > 0
    invalid_json = json.dumps(valid_payload_dict)

    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(raw_output=invalid_json, repair_callback=None, max_attempts=3)

    # Failed on attempt 1, no callback available to retry
    assert len(exc_info.value.errors) == 1


# ============================================================================
# Requirement 4: Unexpected extra fields are rejected
# ============================================================================

def test_unexpected_extra_fields_are_rejected(valid_payload_dict: dict) -> None:
    """Verify that extra unexpected keys trigger a validation error due to extra='forbid'."""
    valid_payload_dict["unauthorized_field"] = "malicious payload"
    invalid_json = json.dumps(valid_payload_dict)

    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(invalid_json, repair_callback=None)

    first_error = exc_info.value.errors[0]
    assert isinstance(first_error, ValidationError)
    error_types = [e["type"] for e in first_error.errors()]
    assert "extra_forbidden" in error_types


# ============================================================================
# Requirement 5: Empty tags are rejected
# ============================================================================

@pytest.mark.parametrize(
    "bad_tags",
    [
        [],               # Empty list (violates min_length=1)
        [""],             # Empty string item
        ["   "],          # Whitespace-only string item
        ["valid", ""],    # Mixed with an empty item
    ],
)
def test_empty_tags_are_rejected(valid_payload_dict: dict, bad_tags: list) -> None:
    """Verify that empty tag lists, empty string tags, or whitespace tags are rejected."""
    valid_payload_dict["tags"] = bad_tags
    invalid_json = json.dumps(valid_payload_dict)

    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(invalid_json, repair_callback=None)

    first_error = exc_info.value.errors[0]
    assert isinstance(first_error, ValidationError)


# ============================================================================
# Requirement 6: Malformed JSON is handled safely
# ============================================================================

@pytest.mark.parametrize(
    "malformed_json_str",
    [
        "{title: unquoted_value}",          # Unquoted key and value
        '{"title": "Unclosed string, ...',  # Unclosed string and missing brace
        "",                                 # Empty string
        "   \n  \t  ",                      # Whitespace only
        "['not', 'a', 'json', 'object']",   # Valid JSON but a list, not dict object
        "12345",                            # Valid JSON primitive integer, not dict object
    ],
)
def test_malformed_json_handled_safely(malformed_json_str: str) -> None:
    """Verify that malformed JSON strings or non-dict structures do not cause unhandled crashes."""
    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(raw_output=malformed_json_str, repair_callback=None, max_attempts=2)

    assert len(exc_info.value.errors) == 1
    # Check that error is either JSONDecodeError or ValueError for wrong top-level type
    assert isinstance(exc_info.value.errors[0], (json.JSONDecodeError, ValueError))


# ============================================================================
# Requirement 7: The repair callback is not called after the final allowed attempt
# ============================================================================

def test_repair_callback_not_called_after_final_attempt() -> None:
    """Verify that with max_attempts=N, the repair callback is called at most N-1 times."""
    tracker = MagicMock(return_value=json.dumps({"bad": "still bad"}))
    invalid_input = "not valid json at all"

    max_attempts = 3
    with pytest.raises(StructuredOutputError):
        validate_with_retry(
            raw_output=invalid_input,
            repair_callback=tracker,
            max_attempts=max_attempts,
        )

    # For 3 attempts:
    # Attempt 1 fails -> tracker called (1st time)
    # Attempt 2 fails -> tracker called (2nd time)
    # Attempt 3 fails -> final allowed attempt, tracker MUST NOT be called!
    assert tracker.call_count == 2


def test_single_attempt_never_calls_repair_callback() -> None:
    """Verify that with max_attempts=1, the repair callback is never called."""
    tracker = MagicMock(return_value="{}")
    invalid_input = "invalid json"

    with pytest.raises(StructuredOutputError):
        validate_with_retry(
            raw_output=invalid_input,
            repair_callback=tracker,
            max_attempts=1,
        )

    assert tracker.call_count == 0


# ============================================================================
# Additional Edge Case & Schema Bounds Tests
# ============================================================================

@pytest.mark.parametrize(
    "field_name, invalid_val",
    [
        ("title", "ab"),                            # Too short (< 3)
        ("title", "a" * 81),                        # Too long (> 80)
        ("summary", "Short"),                       # Too short (< 10)
        ("summary", "s" * 301),                     # Too long (> 300)
        ("priority", "urgent"),                     # Not in ('low', 'medium', 'high')
        ("priority", "NORMAL"),                     # Case sensitive
        ("estimated_hours", 0),                     # Must be > 0
        ("estimated_hours", -1.5),                  # Negative
        ("estimated_hours", 200.1),                 # Exceeds 200
        ("tags", ["t1", "t2", "t3", "t4", "t5", "t6"]), # Exceeds 5 items
    ],
)
def test_schema_constraints_strictly_enforced(
    valid_payload_dict: dict, field_name: str, invalid_val
) -> None:
    """Verify boundary conditions for all individual schema fields."""
    valid_payload_dict[field_name] = invalid_val
    invalid_json = json.dumps(valid_payload_dict)

    with pytest.raises(StructuredOutputError) as exc_info:
        validate_with_retry(invalid_json, repair_callback=None)

    first_err = exc_info.value.errors[0]
    assert isinstance(first_err, ValidationError)


def test_invalid_max_attempts_argument() -> None:
    """Verify max_attempts < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_attempts must be an integer >= 1"):
        validate_with_retry("{}", max_attempts=0)
