from __future__ import annotations

from coverage_support import run_suite_with_coverage


def main() -> int:
    return run_suite_with_coverage(
        label="Full test coverage",
        test_filter=lambda test: True,
        enable_e2e_subprocess_coverage=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
