from __future__ import annotations

from coverage_support import run_suite_with_coverage


def main() -> int:
    return run_suite_with_coverage(
        label="Unit test coverage",
        test_filter=lambda test: not test.__class__.__module__.endswith("_e2e"),
        enable_e2e_subprocess_coverage=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())

