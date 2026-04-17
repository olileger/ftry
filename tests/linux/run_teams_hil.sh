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

ftry pop -t "$REPO_ROOT/samples/teams/hil-grp-feature-debate-team/team.yaml" -p "Should we add live match alerts to our football fan mobile app this quarter?"
ftry pop -t "$REPO_ROOT/samples/teams/hil-han-support-routing-team/team.yaml" -p "I was charged twice for my football premium subscription and I need help."
ftry pop -t "$REPO_ROOT/samples/teams/hil-mag-launch-planning-team/team.yaml" -p "We are launching a weekly football digest for coaches next month. Build a lightweight launch brief."
ftry pop -t "$REPO_ROOT/samples/teams/hil-seq-support-brief-team/team.yaml" -p "Customer says the football match report export failed."
