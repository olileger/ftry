---
name: python-windows-agent-cli
description: Guidance for developing Python applications on Windows in this repository, including latest-stable Python practices, robust CLI design, and mandatory use of Microsoft Agent Framework for agentic features.
---

# Python, Windows, Agent Framework, and CLI Guidance

Use this skill when working on:

- Python code in this repository;
- Windows-specific development or execution details;
- agentic application design;
- command-line interfaces, terminal UX, and automation behavior.

## Required baseline

- Target the **latest stable Python version available on Windows**.
- The current reference baseline is **Python 3.14.3**.
- Prefer the official **Python Install Manager** on Windows.
- Always use a project-local virtual environment named **`.venv`**.
- Prefer **PowerShell** commands and examples.

Recommended local setup:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Mandatory framework for agentic features

For any agentic capability in this repository, you must **use Microsoft Agent Framework for Python** in the latest compatible version.

```powershell
pip install --upgrade agent-framework
```

Rules:

- Use **Agents** for conversational or open-ended behaviors.
- Use **Workflows** for deterministic multi-step orchestration.
- If a regular function is enough, prefer a regular function instead of an agent.
- Do not assume `.env` is loaded automatically; explicitly load it at startup if needed.
- Keep model client setup, agent or workflow definitions, tool registration, configuration loading, and CLI entrypoints clearly separated.
- Keep prompts, tool contracts, and instructions explicit and reviewable.
- Treat observability, tool safety, and configuration hygiene as first-class concerns.

## Python engineering practices

- Use **type hints** for public functions, methods, and important internal flows.
- Prefer **small, focused functions and modules**.
- Prefer **`pathlib.Path`** over manual path string manipulation.
- Prefer `dataclass`, `Enum`, `TypedDict`, and `Protocol` when they improve clarity.
- Avoid duplication; extract shared logic into helpers.
- Avoid magic values; use constants, settings, and environment variables.
- Use **UTF-8** consistently for file and console I/O.
- Use **timezone-aware datetimes**.
- Keep side effects at the edges: filesystem, network, environment variables, subprocesses, and console rendering.
- Handle failures explicitly; do not silently swallow exceptions.
- Convert domain errors into clear CLI-facing messages at the application boundary.
- Prefer standard-library solutions when they are sufficient.

## Windows-specific guidance

- Assume execution happens in **Windows PowerShell** unless the task says otherwise.
- Use **Windows-style paths** in user-facing examples.
- In code, still prefer **`pathlib`** for robust path handling.
- Validate paths before use and return actionable error messages.
- Be careful with quoting, spaces in paths, and OneDrive-synced directories.
- Support `Ctrl+C` cleanly.
- Avoid Unix-only shell assumptions.

## CLI implementation rules

Any CLI built here must be:

- easy to use interactively;
- predictable in scripts and CI;
- robust in interactive and non-interactive terminals.

### CLI architecture

- Create a dedicated CLI entrypoint with a `main()` function returning an `int`.
- Use:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- Prefer **`argparse`** by default.
- Use subcommands when the tool exposes multiple actions.
- Separate argument parsing, application logic, and display rendering.
- If async work is needed, keep the sync CLI boundary small and use `asyncio.run(...)`.

### Input handling

- Support explicit command-line arguments first.
- Support **stdin** when piping or automation makes sense.
- Detect terminal interactivity with `sys.stdin.isatty()` and `sys.stdout.isatty()`.
- Never block on prompts in non-interactive mode.
- For destructive actions, use interactive confirmation plus a `--yes` override for automation.
- Validate and normalize inputs early before passing them to core logic.

### Output and display

- Write normal results to **stdout**.
- Write errors, warnings, and diagnostics to **stderr**.
- Keep default output stable and script-friendly.
- If enhanced rendering is useful, prefer **`rich`** for tables, colors, progress, and summaries.
- Rich output must remain optional behavior, not a hard dependency for correctness.
- Offer machine-readable output such as `--json` when useful.
- Avoid noisy output in normal mode.

### Common CLI options

Include these when relevant:

- `--help`
- `--version`
- `--verbose`
- `--quiet`
- `--json`
- `--yes`

### Errors, exits, and logging

- Return `0` on success and non-zero exit codes on failure.
- Keep error messages concise and actionable.
- Catch exceptions at the CLI boundary when converting them into user-facing messages.
- Reserve stack traces for debug or verbose mode.
- Use the standard `logging` module.
- Map verbosity flags to logging levels.
- Do not mix logs with structured stdout output.

## Suggested project layout

```text
src/
  app/
    cli.py
    config.py
    agents/
    workflows/
    tools/
    services/
    ui/
tests/
```

Use this layout intent:

- `cli.py`: argument parsing and process exit handling;
- `config.py`: settings, environment loading, and validation;
- `agents/`: Microsoft Agent Framework agent definitions;
- `workflows/`: deterministic orchestration logic;
- `tools/`: tools exposed to agents;
- `services/`: business logic and integrations;
- `ui/`: prompts, renderers, tables, and terminal presentation helpers.

## Dependency guidance

- Keep dependencies minimal.
- Use **Microsoft Agent Framework** for agentic behavior.
- Use **`python-dotenv`** only when explicit `.env` loading is needed.
- Use **`rich`** when terminal UX benefits from enhanced rendering.
- Avoid overlapping libraries with duplicated responsibilities unless there is a strong reason.

## Non-negotiable rules

- Python work in this repository targets the latest stable Windows-compatible baseline.
- Agentic features must use **Microsoft Agent Framework for Python**.
- CLI code must be testable, scriptable, and Windows-friendly.
- Input handling, output behavior, logging, and error handling must be deliberate and consistent.
