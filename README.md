# ftry

![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)

Minimal CLI named `ftry`.

## Available commands

For now, the commands are mocks, except for `line` and `pop`.

- `ftry build`
- `ftry break`
- `ftry pop`
- `ftry land`
- `ftry line`

Example:

```powershell
ftry build
```

Output:

```text
build
```

The `ftry line` command loads its output from `src\ftry\line.txt`. To change the visual output, simply edit that file.

The `ftry pop` command loads either an agent (`-a`) or a team of agents (`-t`) from a YAML file, sends the prompt passed with `-p`, then displays the model response.
In an interactive terminal, it plays a slightly longer neon ASCII `POP` skateboard animation before the run starts, loading the static `POP` banner from `src\ftry\pop.txt`, reusing the skateboard style from `src\ftry\line.txt`, and leaving the final frame visible on screen.
For direct agent runs (`ftry pop -a`), `ftry` now keeps the same Agent Framework session across turns and lets the model decide, via a structured turn status, whether the conversation is done or whether it is waiting for the next user input.
For team runs (`ftry pop -t`), `ftry` now infers both the orchestration type and the need for Human in the Loop `request_info` from the current user request, the team prompt, and the participating agent roles, then resumes supported Microsoft Agent Framework workflows when those request events are emitted.

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

From the project root, install the CLI in editable local mode:

```powershell
python -m pip install -e .
```

Then the command is available in the terminal:

```powershell
ftry break
```

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
