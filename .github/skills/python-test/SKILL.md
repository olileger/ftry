---
name: python-test
description: Guidance for designing, writing, running, and debugging Python tests in this repository on Windows.
---

# Python Testing Guidance

Use this skill when working on:

- unit tests, integration tests, and regression tests for Python code;
- test structure, fixtures, mocks, parametrization, and edge-case coverage;
- debugging failing Python tests on Windows;
- improving confidence in Python CLI, services, workflows, and agentic components.

## Baseline expectations

- Follow the repository Python baseline from the existing Windows/Python skill.
- Use the project-local virtual environment named **`.venv`**.
- Prefer **PowerShell** commands and examples.
- Keep tests deterministic, isolated, and fast by default.

Typical setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip pytest
pytest
```

## Preferred testing approach

- Prefer **`pytest`** for new Python tests unless the surrounding code already uses another framework.
- Keep application code and test code clearly separated.
- Put tests under a top-level **`tests/`** directory unless the repository already uses a different layout.
- Name test files `test_*.py`.
- Name test functions `test_*`.
- Group tests by behavior, not by implementation detail.

## What good tests should cover

- expected success paths;
- meaningful edge cases;
- invalid inputs and explicit error behavior;
- Windows-specific path, quoting, and filesystem scenarios when relevant;
- CLI exit codes, stdout, and stderr behavior for command-line tools;
- agentic workflow boundaries, tool contracts, and deterministic orchestration logic when applicable.

## Test design rules

- Prefer small, focused tests with one clear reason to fail.
- Use fixtures to remove duplication, not to hide important setup.
- Prefer parametrized tests over repetitive copy-paste cases.
- Mock only true external boundaries such as network, clock, environment, subprocesses, and filesystem writes when isolation is needed.
- Do not mock so deeply that tests stop validating real behavior.
- Assert observable behavior and stable contracts.
- Avoid assertions that depend on fragile internal implementation details unless that detail is the contract being tested.

## CLI and Windows-specific testing

When testing CLI behavior:

- verify exit status separately from output content;
- assert **stdout** for normal output and **stderr** for errors or diagnostics;
- cover non-interactive behavior when input is piped or terminal prompts are unavailable;
- test paths with spaces and Windows-style separators when path handling matters;
- prefer `pathlib.Path` and temporary directories over hard-coded paths.

## Agentic and workflow testing

For Python code using Microsoft Agent Framework:

- keep agent instructions and tool contracts explicit so they can be tested;
- test deterministic workflow steps as regular Python logic where possible;
- isolate model-dependent behavior behind boundaries that can be substituted in tests;
- prefer contract tests for tools and adapters over brittle prompt-string assertions;
- validate failure handling for tool errors, invalid configuration, and missing environment settings.

## Useful pytest patterns

### Parametrization

```python
import pytest


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("42", 42),
        ("0042", 42),
    ],
)
def test_parse_count(raw_value: str, expected: int) -> None:
    assert parse_count(raw_value) == expected
```

### Error assertions

```python
import pytest


def test_parse_count_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        parse_count("-1")
```

### Temporary filesystem usage

```python
def test_write_report_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "report.txt"

    write_report(output_path, "ok")

    assert output_path.read_text(encoding="utf-8") == "ok"
```

## Debugging failing tests

- Reproduce the smallest failing scope first.
- Run a single file, class, or test before rerunning the full suite.
- Add or improve assertions rather than printing excessively.
- Prefer fixing nondeterminism at the root cause instead of adding retries.
- If a test only fails on Windows, inspect path normalization, newline handling, encoding, shell quoting, and file locking assumptions.

Useful commands:

```powershell
pytest tests\test_example.py
pytest tests\test_example.py -k specific_case -vv
pytest -x
pytest --maxfail=1
```

## Non-negotiable rules

- New Python tests should be readable, deterministic, and maintainable.
- Test names must describe behavior clearly.
- Failures must be actionable.
- Prefer targeted tests that protect behavior over broad but vague coverage.
