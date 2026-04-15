#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
ACTIVATE_SH="$REPO_ROOT/.venv/bin/activate"

if [ ! -f "$ACTIVATE_SH" ]; then
    echo "Virtual environment not found at \"$ACTIVATE_SH\"." >&2
    exit 1
fi

# shellcheck disable=SC1090
. "$ACTIVATE_SH"

ftry pop -t "$REPO_ROOT/samples/teams/better-prompt/team.yaml" -p "Write a short LinkedIn post announcing a new football stats app."
ftry pop -t "$REPO_ROOT/samples/teams/con-release-readiness-team/team.yaml" -p "We are launching a football stats app for amateur clubs next week. Give us a quick release readiness view."
ftry pop -t "$REPO_ROOT/samples/teams/grp-feature-debate-team/team.yaml" -p "Should we add live match alerts to our football fan mobile app this quarter?"
ftry pop -t "$REPO_ROOT/samples/teams/han-support-routing-team/team.yaml" -p "I was charged twice for my football premium subscription and I need help."
ftry pop -t "$REPO_ROOT/samples/teams/mag-launch-planning-team/team.yaml" -p "We are launching a weekly football digest for coaches next month. Build a lightweight launch brief."
ftry pop -t "$REPO_ROOT/samples/teams/seq-support-brief-team/team.yaml" -p "Customer says the football match report export failed twice and wants a clear status update today."
