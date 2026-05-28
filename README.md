# ftry

![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)

Minimal CLI named `ftry`.

## Available commands

- `ftry build`
- `ftry pop`
- `ftry line`

The `ftry build` command takes a prompt with `-p`, uses an internal builder team to decide whether the request should become one agent or one team, then writes the generated YAML files under `.\output\<generated-name>\`.
When MCP integration is relevant, `build` reuses descriptors already present in `.\mcp\` and can generate new conservative MCP descriptors directly inside the generated output directory as `mcp-*.yaml` files when the request clearly requires them.

Example:

```powershell
ftry build -p "Create an agent that drafts concise release notes from raw product updates."
```

The `ftry line` command loads its output from `src\ftry\line.txt`. To change the visual output, simply edit that file.

The `ftry pop` command loads either an agent (`-a`) or a team of agents (`-t`) from a YAML file, sends the prompt passed with `-p`, then displays the model response.
In an interactive terminal, it prints the static `POP` banner before the run starts, loading it from `src\ftry\ascii-art\pop.txt`.
For direct agent runs (`ftry pop -a`), `ftry` now keeps the same Agent Framework session across turns and lets the model decide, via a structured turn status, whether the conversation is done or whether it is waiting for the next user input.
For team runs (`ftry pop -t`), `ftry` now infers both the orchestration type and the need for Human in the Loop `request_info` from the current user request, the team prompt, and the participating agent roles, then resumes supported Microsoft Agent Framework workflows when those request events are emitted.
When an agent or team YAML declares `mcp`, `pop` resolves those descriptors from `mcp-*.yaml` files next to the loaded YAML first, then from a sibling `.\mcp\` folder next to the loaded YAML, then from the parent directory's `.\mcp\` folder, then falls back to the current working directory `.\mcp\` folder, and wires them into Microsoft Agent Framework MCP tools at runtime.

Example:

```
ftry pop -a .\samples\agents\poete.yaml -p "Write a poem about rain"
```

Team example:

```powershell
ftry pop -t .\samples\teams\better-prompt\team.yaml -p "Build a better prompt to summarize this text"
```

Group chat example:

```powershell
ftry pop -t .\samples\teams\grp-feature-debate-team\team.yaml -p "We want a reminder feature that nudges users before a payment is due."
```

Handoff example:

```powershell
ftry pop -t .\samples\teams\han-support-routing-team\team.yaml -p "I was charged twice for my monthly plan and I need a short refund update I can send to the customer."
```

Magentic example:

```powershell
ftry pop -t .\samples\teams\mag-launch-planning-team\team.yaml -p "We are launching a weekly digest feature for project managers next month. Build a lightweight rollout brief with the goal, main risks, and next action."
```

Human-in-the-Loop team examples:

```powershell
ftry pop -t .\samples\teams\hil-seq-support-brief-team\team.yaml -p "Customer cannot log in since yesterday. We restarted the auth service. We are still checking the root cause."
ftry pop -t .\samples\teams\hil-han-support-routing-team\team.yaml -p "I cannot access the workspace after the subscription change and I need a customer-ready update."
ftry pop -t .\samples\teams\hil-grp-feature-debate-team\team.yaml -p "We want to add GenAI to our chatbot."
ftry pop -t .\samples\teams\hil-mag-launch-planning-team\team.yaml -p "We want to launch an AI weekly digest for team leads next month. Build a lightweight rollout brief."
```

## Local installation

Prerequisites:

- Python 3.10 or newer
- `pip`

On Windows ARM64, the project currently pins `cryptography` to `<=46.0.3` because newer releases are not always published with a compatible wheel, which can force a local Rust/MSVC build and make `pip install -e .` fail.

From the project root, install the CLI in editable local mode:

```powershell
python -m pip install -e .
```

Then the command is available in the terminal:

```powershell
ftry line
```

## MCP descriptors

Reusable MCP server descriptors can live either next to the loaded agent/team YAML as local `mcp-*.yaml` files, under a neighboring `.\mcp\` folder, or in the current working directory under `.\mcp\`.

Example descriptor:

```yaml
name: "file-system"
description: |
  Local filesystem access for workspace operations.
transport: "stdio"
command: "uvx"
args:
  - "mcp-server-filesystem"
  - "C:\\work"
```

Supported descriptor fields:

- `name`
- `description` (optional)
- `transport`: `stdio`, `http`, or `websocket`
- `allowed-tools` (optional)
- `command`, `args`, optional `env` for `stdio`
- `url`, optional `headers` for `http` and `websocket`

Secrets must stay out of generated YAML files. Use `env:VAR_NAME` references inside `env` or `headers` values:

```yaml
headers:
  Authorization: "env:GITHUB_TOKEN"
```

Agent and team YAML files reference MCP descriptors by name:

```yaml
name: "Workspace Agent"
model:
  name: "gpt-4o-2024-08-06"
  provider: "openai"
  api-key: "env:OAI_API_KEY"
mcp:
  - "file-system"
prompt: |
  Help the user work with local files.
```

Teams may also declare shared MCP descriptors at the top level and extra MCP descriptors per participant entry:

```yaml
name: "Launch Team"
mcp:
  - "shared-files"
agents:
  - file: .\agent-scope.yaml
    mcp:
      - "scope-search"
prompt: |
  Coordinate the work.
```

To run MCP-enabled agents or teams, install the project dependencies so the Python `mcp` package is available:

```powershell
python -m pip install -e .
```

If you already had `ftry` installed before MCP support was added, reinstall it so the new dependency is picked up.

At `ftry pop` runtime, MCP-enabled agents and teams now connect to their referenced MCP servers before first use and discover the live capability directory exposed by those servers. This live discovery stays in memory only and is injected into the agent execution context for the current run; `ftry` does not write any discovery snapshot back into YAML or cache files.

`ftry build` follows a conservative MCP policy:

- reuse an existing descriptor from `.\mcp\` whenever possible;
- only create a new descriptor when the request clearly requires a new server, and place generated descriptors next to the generated `agent.yaml` or `team.yaml` as `mcp-*.yaml` files;
- never write clear-text secrets;
- never invent runnable stdio commands or placeholder HTTP/WebSocket URLs; reuse a real descriptor or provide real connection details only;
- when a required MCP connection detail is missing, `build` asks for that exact command or URL instead of generating incomplete YAML.

## Tests

Install the test dependencies:

```powershell
python -m pip install -e .[test]
```

Run the unit tests with coverage displayed at the end:

```powershell
.\tests\windows\unit.bat
```

```sh
./tests/linux/unit.sh
```

Run the CLI end-to-end tests with coverage:

```powershell
.\tests\windows\e2e.bat
```

```sh
./tests/linux/e2e.sh
```

Run the full suite with combined coverage (unit + end-to-end):

```powershell
.\tests\windows\all.bat
```

```sh
./tests/linux/all.sh
```

Run the full suite without coverage:

```powershell
python -m unittest discover -s tests -q
```
