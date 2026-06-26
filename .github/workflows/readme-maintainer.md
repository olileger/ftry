---
emoji: 📘
description: Keep README.md aligned with feature changes introduced by commits on main
on:
  push:
    branches: [main]
    paths-ignore:
      - README.md
  workflow_dispatch:
  roles: all
permissions:
  contents: read
  issues: read
  pull-requests: read
checkout:
  fetch-depth: 0
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
  edit:
safe-outputs:
  create-pull-request:
    title-prefix: "[docs] "
    branch-prefix: "agent/readme-maintainer/"
    labels: [documentation, automation]
    draft: false
    if-no-changes: ignore
    allowed-files:
      - README.md
---

# README Maintainer

## Task

Keep `README.md` accurate after repository commits.

When triggered by a push, inspect the triggering commit range and understand the user-facing feature impact. Focus on behavior, commands, options, examples, setup requirements, or documented workflows that changed. Ignore purely internal refactors, tests, formatting, generated files, or changes that do not affect what a README reader needs to know.

Compare the current `README.md` with the feature impact of the commit range:

1. If `README.md` already reflects the commit, call `noop` with a short explanation.
2. If documentation is missing or stale, update only `README.md`.
3. Keep the README concise and consistent with its existing structure and tone.
4. Do not document speculative behavior; only document behavior supported by the committed code.
5. If several commits are included in the push, summarize them from the feature perspective and make one coherent README update.

Use `gh` through the configured GitHub tool for repository reads when needed. Useful commands include:

```bash
gh api repos/${GITHUB_REPOSITORY}/commits/${GITHUB_SHA}
gh api repos/${GITHUB_REPOSITORY}/compare/${{ github.event.before }}...${{ github.event.after }}
```

For manual runs, inspect the latest commits on the checked-out branch and apply the same README freshness check.

## Safe Outputs

- Use `create-pull-request` only when `README.md` needs an update.
- Call `noop` when the README is already accurate or when the commit has no README-relevant feature impact.
