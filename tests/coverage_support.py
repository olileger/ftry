from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
COVERAGE_DATA_FILE = REPO_ROOT / ".coverage"


def ensure_src_on_path() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


def iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def discover_tests() -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    return loader.discover(
        start_dir=str(REPO_ROOT / "tests" / "src"),
        pattern="test_*.py",
        top_level_dir=str(REPO_ROOT),
    )


def cleanup_coverage_files() -> None:
    for path in REPO_ROOT.glob(".coverage*"):
        if path.is_file():
            path.unlink(missing_ok=True)


def run_suite_with_coverage(
    *,
    label: str,
    test_filter: Callable[[unittest.case.TestCase], bool],
    enable_e2e_subprocess_coverage: bool,
) -> int:
    try:
        from coverage import Coverage
    except ImportError:
        print(
            "Coverage support is not installed. Run `python -m pip install -e .[test]` first.",
            file=sys.stderr,
        )
        return 1

    ensure_src_on_path()
    cleanup_coverage_files()

    previous_e2e_coverage = os.environ.get("FTRY_E2E_COVERAGE")
    previous_coverage_file = os.environ.get("COVERAGE_FILE")
    if enable_e2e_subprocess_coverage:
        os.environ["FTRY_E2E_COVERAGE"] = "1"
        os.environ["COVERAGE_FILE"] = str(COVERAGE_DATA_FILE)
    else:
        os.environ.pop("FTRY_E2E_COVERAGE", None)

    coverage = Coverage(source=["ftry"], data_file=str(COVERAGE_DATA_FILE), data_suffix=True)
    try:
        coverage.start()
        discovered = discover_tests()
        filtered_suite = unittest.TestSuite(test for test in iter_tests(discovered) if test_filter(test))
        result = unittest.TextTestRunner(verbosity=1).run(filtered_suite)
        coverage.stop()
        coverage.save()
    finally:
        if previous_e2e_coverage is None:
            os.environ.pop("FTRY_E2E_COVERAGE", None)
        else:
            os.environ["FTRY_E2E_COVERAGE"] = previous_e2e_coverage

        if previous_coverage_file is None:
            os.environ.pop("COVERAGE_FILE", None)
        else:
            os.environ["COVERAGE_FILE"] = previous_coverage_file

    combined = Coverage(source=["ftry"], data_file=str(COVERAGE_DATA_FILE))
    combined.combine()

    print(f"\n{label}:")
    combined.report(show_missing=True)
    return 0 if result.wasSuccessful() else 1

