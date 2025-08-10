import sys
import os
import argparse
from pathlib import Path
import pytest


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="BDD Test Automation Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests in QA environment
  python test_runner.py --env qa

  # Run smoke tests in DEV environment with detailed report
  python test_runner.py --env dev --tags smoke --report

  # Run specific feature file
  python test_runner.py --env qa --feature features/complaint_capture.feature
        """
    )

    parser.add_argument(
        "--env",
        default="qa",
        choices=["dev", "qa", "staging", "production"],
        help="Environment to run tests against (default: qa)"
    )

    parser.add_argument(
        "--tags",
        help="Pytest markers to run (e.g., 'smoke', 'regression', 'smoke and not slow')"
    )

    parser.add_argument(
        "--feature",
        help="Specific feature file or directory to run"
    )

    parser.add_argument(
        "--report",
        action="store_true",
        default=True,
        help="Generate HTML report after test run (default: True)"
    )

    parser.add_argument(
        "--report-title",
        default="BDD Test Automation Report",
        help="Custom title for the HTML report"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel using N workers"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity (use -vv for more verbose)"
    )

    parser.add_argument(
        "--capture",
        choices=["yes", "no", "sys"],
        default="no",
        help="Capture stdout/stderr (default: no)"
    )

    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Set environment variable for configuration loading
    os.environ["ENVIRONMENT"] = args.env
    print(f"\n{'='*60}")
    print(f"Starting BDD Test Execution")
    print(f"Environment: {args.env.upper()}")
    print(f"{'='*60}\n")

    # Build pytest command
    pytest_args = []

    # Add test directory or specific feature
    if args.feature:
        pytest_args.append(args.feature)
    else:
        # Execute all feature files in the 'features' directory
        pytest_args.append("bdd_tests/features/")

    # Add verbosity
    pytest_args.append("-" + "v" * args.verbose)

    # Add capture setting
    pytest_args.append(f"--capture={args.capture}")

    # Add environment
    pytest_args.extend(["--env", args.env])

    # Add tags/markers if specified
    if args.tags:
        pytest_args.extend(["-m", args.tags])

    # Add parallel execution if specified
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])

    # Add HTML report generation
    if args.report:
        report_name = f"reports/test_report_{args.env}_{Path(__file__).parent.name}.html"
        pytest_args.extend([
            "--html", report_name,
            "--self-contained-html",
            f"--html-report-title={args.report_title}"
        ])

    # Add logging
    pytest_args.extend([
        "--log-cli-level=INFO",
        "--log-cli-format=%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
        "--log-cli-date-format=%Y-%m-%d %H:%M:%S"
    ])

    # Add any additional pytest arguments
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    # Print command for debugging
    print(f"Executing: pytest {' '.join(pytest_args)}\n")

    # Run pytest
    exit_code = pytest.main(pytest_args)

    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed successfully!")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    print(f"{'='*60}\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
