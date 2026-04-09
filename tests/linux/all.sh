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
python "$REPO_ROOT/tests/run_all_tests.py" "$@"
