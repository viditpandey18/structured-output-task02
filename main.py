"""Task 02 — Structured Output

This module implements strict JSON schema validation, retry, and deterministic
repair for AI-generated outputs using Pydantic v2.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# ============================================================================
# 1. Strict Schema Definition
# ============================================================================

class TaskResult(BaseModel):
    """Strict schema for AI-generated task breakdown results.

    Validation rules:
    - title: 3 to 80 characters
    - summary: 10 to 300 characters
    - priority: strictly 'low', 'medium', or 'high'
    - estimated_hours: strictly greater than 0 and at most 200
    - tags: list of 1 to 5 non-empty strings
    - Extra fields: strictly forbidden
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    title: str = Field(
        ...,
        min_length=3,
        max_length=80,
        description="Concise name of the task (3-80 chars)",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=300,
        description="Detailed description of the task (10-300 chars)",
    )
    priority: Literal["low", "medium", "high"] = Field(
        ...,
        description="Priority level: 'low', 'medium', or 'high'",
    )
    estimated_hours: float = Field(
        ...,
        gt=0,
        le=200,
        description="Estimated completion time in hours (0 < hours <= 200)",
    )
    tags: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="List of 1 to 5 descriptive task tags",
    )

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, tags: list[str]) -> list[str]:
        """Ensure tags list is non-empty and contains no empty or whitespace strings."""
        if not tags:
            raise ValueError("Tags list cannot be empty; must contain 1 to 5 tags.")
        cleaned_tags: list[str] = []
        for index, tag in enumerate(tags):
            if not isinstance(tag, str):
                raise TypeError(f"Tag at index {index} must be a string, got {type(tag).__name__}.")
            stripped = tag.strip()
            if not stripped:
                raise ValueError(f"Tag at index {index} cannot be empty or whitespace-only.")
            cleaned_tags.append(stripped)
        return cleaned_tags


# ============================================================================
# 2. Custom Exception
# ============================================================================

class StructuredOutputError(Exception):
    """Raised when output fails validation and cannot be repaired within max attempts."""

    def __init__(self, message: str, errors: Optional[list[Exception]] = None) -> None:
        super().__init__(message)
        self.errors: list[Exception] = errors or []


# ============================================================================
# 3. Validation and Retry Logic
# ============================================================================

def validate_with_retry(
    raw_output: str,
    repair_callback: Optional[Callable[[str, Exception], str]] = None,
    max_attempts: int = 3,
) -> TaskResult:
    """Validates raw JSON string output against the TaskResult schema with retry/repair.

    Args:
        raw_output: The raw text/JSON string from an AI model.
        repair_callback: Optional callable (current_output, error) -> repaired_output.
        max_attempts: Maximum number of attempts allowed (must be >= 1).

    Returns:
        TaskResult: The validated Pydantic model instance.

    Raises:
        ValueError: If max_attempts < 1 or raw_output is not a string.
        StructuredOutputError: If all attempts fail validation.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be an integer >= 1.")
    if not isinstance(raw_output, str):
        raise ValueError(f"raw_output must be a string, got {type(raw_output).__name__}.")

    current_output = raw_output
    collected_errors: list[Exception] = []

    for attempt in range(1, max_attempts + 1):
        try:
            # Step 1: Parse JSON safely using Python's json standard library module
            parsed_data = json.loads(current_output)

            # Ensure parsed data is a JSON object / dict, not a bare primitive or list
            if not isinstance(parsed_data, dict):
                raise ValueError(
                    f"Expected top-level JSON object (dict), but received {type(parsed_data).__name__}."
                )

            # Step 2: Validate parsed data strictly with Pydantic
            validated_result = TaskResult.model_validate(parsed_data)
            return validated_result

        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as err:
            collected_errors.append(err)

            # Check if there are attempts remaining AND a repair callback is provided
            if attempt < max_attempts and repair_callback is not None:
                try:
                    current_output = repair_callback(current_output, err)
                except Exception as cb_err:
                    raise StructuredOutputError(
                        f"Repair callback failed on attempt {attempt}: {cb_err}",
                        errors=collected_errors,
                    ) from cb_err
            else:
                # Stop: No remaining attempts or no repair callback available
                break

    # If all attempts fail, raise a clear custom exception with complete error context
    last_err = collected_errors[-1] if collected_errors else "Unknown error"
    raise StructuredOutputError(
        f"Validation failed after {len(collected_errors)} attempt(s). Last error: {last_err}",
        errors=collected_errors,
    )


# ============================================================================
# 4. Deterministic Mock Repair Callback (Offline Demo)
# ============================================================================

def mock_repair_callback(invalid_output: str, error: Exception) -> str:
    """Deterministic mock repair callback that repairs common LLM output mistakes.

    Repairs handled:
    - Stripping markdown code fences (```json ... ```)
    - Normalising invalid priority levels (e.g. 'urgent' / 'critical' -> 'high')
    - Clamping out-of-bounds estimated_hours (e.g. >200 or <=0)
    - Removing extra disallowed fields
    - Supplying default valid tags when tags list is empty or invalid
    - Truncating or expanding strings to satisfy min/max constraints
    """
    cleaned = invalid_output.strip()

    # 1. Remove markdown fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()

    # 2. Attempt to parse JSON; if malformed, extract JSON-like object or construct fallback
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Simple heuristic to balance missing braces / quotes or extract substring
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    if not isinstance(data, dict):
        data = {}

    # 3. Repair / sanitize fields to satisfy TaskResult schema
    # Title (3 to 80 chars)
    raw_title = str(data.get("title", "Refactor authentication flow"))
    if len(raw_title) < 3:
        raw_title = "Task: " + raw_title if len("Task: " + raw_title) >= 3 else "Default Task Title"
    elif len(raw_title) > 80:
        raw_title = raw_title[:80]
    data["title"] = raw_title

    # Summary (10 to 300 chars)
    raw_summary = str(data.get("summary", "Complete task implementation and unit test coverage."))
    if len(raw_summary) < 10:
        raw_summary = "Detailed description: " + raw_summary
    if len(raw_summary) > 300:
        raw_summary = raw_summary[:300]
    data["summary"] = raw_summary

    # Priority ('low', 'medium', 'high')
    raw_priority = str(data.get("priority", "medium")).lower()
    if raw_priority in {"critical", "urgent", "highest", "immediate"}:
        data["priority"] = "high"
    elif raw_priority not in {"low", "medium", "high"}:
        data["priority"] = "medium"
    else:
        data["priority"] = raw_priority

    # Estimated hours (0 < hours <= 200)
    try:
        hours = float(data.get("estimated_hours", 4.0))
        if hours <= 0:
            hours = 1.0
        elif hours > 200:
            hours = 40.0
        data["estimated_hours"] = hours
    except (ValueError, TypeError):
        data["estimated_hours"] = 4.0

    # Tags (1 to 5 non-empty strings)
    raw_tags = data.get("tags")
    if not isinstance(raw_tags, list) or len(raw_tags) == 0:
        data["tags"] = ["engineering", "backend"]
    else:
        cleaned_tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        if not cleaned_tags:
            cleaned_tags = ["general"]
        data["tags"] = cleaned_tags[:5]

    # Reject / delete any extra fields
    allowed_fields = {"title", "summary", "priority", "estimated_hours", "tags"}
    extra_keys = [k for k in data if k not in allowed_fields]
    for k in extra_keys:
        del data[k]

    return json.dumps(data)


# ============================================================================
# 5. Interactive / Demonstration Runner
# ============================================================================

def main() -> None:
    """Demonstrates structured output validation, failure detection, and repair."""
    print("=" * 70)
    print(" TASK 02: STRUCTURED OUTPUT VALIDATION & REPAIR DEMO")
    print("=" * 70)

    # Simulated flawed AI output:
    # 1. Wrapped in Markdown code fences
    # 2. Priority is 'critical' (invalid, only low/medium/high permitted)
    # 3. Estimated hours is 250 (exceeds max 200)
    # 4. Tags is empty [] (schema requires 1-5 non-empty tags)
    # 5. Extra disallowed field 'internal_thoughts'
    flawed_ai_output = """```json
{
  "title": "Migrate database schema to PostgreSQL 16",
  "summary": "Execute partitioned table migration with zero downtime.",
  "priority": "critical",
  "estimated_hours": 250,
  "tags": [],
  "internal_thoughts": "Make sure to check connection pool size before running"
}
```"""

    print("\n[STEP 1] Raw output from simulated AI model:")
    print("-" * 50)
    print(flawed_ai_output)
    print("-" * 50)

    # Tracker to observe callback invocation
    callback_invoked = False

    def monitored_repair(output: str, err: Exception) -> str:
        nonlocal callback_invoked
        callback_invoked = True
        print(f"\n[STEP 2] Validation caught error: {err.__class__.__name__}")
        print(f"         Details: {err}")
        print("         -> Invoking deterministic repair callback...")
        repaired = mock_repair_callback(output, err)
        print("         -> Repaired JSON candidate produced:")
        print(f"            {repaired}")
        return repaired

    print("\n[STEP 3] Running validate_with_retry(max_attempts=3)...")
    try:
        result: TaskResult = validate_with_retry(
            raw_output=flawed_ai_output,
            repair_callback=monitored_repair,
            max_attempts=3,
        )

        print("\n[STEP 4] Validation Result:")
        print("+" * 50)
        print(f"Status              : SUCCESS")
        print(f"Is TaskResult class : {isinstance(result, TaskResult)}")
        print(f"Repair was called   : {callback_invoked}")
        print(f"Final Model Object  : {repr(result)}")
        print("\nFinal Validated JSON:")
        print(result.model_dump_json(indent=2))
        print("+" * 50)

    except StructuredOutputError as e:
        print(f"\nValidation failed as expected when not repairable: {e}")


if __name__ == "__main__":
    main()
