import sys
import os
import argparse
import subprocess
from pathlib import Path
import pytest


def ensure_playwright_installed():
    """Ensure Playwright browsers are installed."""
    try:
        import playwright
        # Check if browsers are installed by trying to run a simple command
        result = subprocess.run(
            ["playwright", "install", "--help"], 
            capture_output=True, 
            text=True
        )
        if result.returncode != 0:
            print("Installing Playwright browsers...")
            subprocess.run(["playwright", "install"], check=True)
            print("Playwright browsers installed successfully.")
    except (ImportError, subprocess.CalledProcessError) as e:
        print(f"Error with Playwright installation: {e}")
        print("Please run: pip install playwright && playwright install")
        sys.exit(1)


def main():
    """Main entry point for Playwright-based test runner."""
    parser = argparse.ArgumentParser(
        description="BDD Test Automation Framework Runner - Playwright Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests in QA environment
  python test_runner.py --env qa

  # Run smoke tests in DEV environment with detailed report
  python test_runner.py --env dev --tags smoke --report

  # Run specific feature file
  python test_runner.py --env qa --feature features/complaint_capture.feature
  
  # Run with API timeout override
  python test_runner.py --env qa --api-timeout 60

  # Run tests in parallel
  python test_runner.py --env qa --parallel 4
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
        default="BDD Test Automation Report - Complaints AI Chatbot",
        help="Custom title for the HTML report"
    )

    parser.add_argument(
        "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel using N workers"
    )

    parser.add_argument(
        "--api-timeout",
        type=int,
        metavar="SECONDS",
        help="Override API timeout in seconds"
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
        "--browser-channel",
        default="chrome",
        choices=["chrome", "msedge", "chromium"],
        help="Browser channel for Playwright (default: chrome, though only used for API context)"
    )

    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run in headed mode (for debugging, not applicable to API tests)"
    )

    parser.add_argument(
        "--slowmo",
        type=int,
        default=0,
        help="Slow down operations by N milliseconds (for debugging)"
    )

    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Ensure Playwright is properly installed
    ensure_playwright_installed()

    # Set environment variable for configuration loading
    os.environ["ENVIRONMENT"] = args.env
    print(f"\n{'='*60}")
    print(f"Starting Playwright-based BDD Test Execution")
    print(f"Environment: {args.env.upper()}")
    print(f"Framework: pytest-bdd + Playwright")
    print(f"{'='*60}\n")

    # Build pytest command
    pytest_args = []

    # Add test directory or specific feature
    if args.feature:
        pytest_args.append(args.feature)
    else:
        # Execute all feature step files
        pytest_args.append("features/steps/")

    # Add verbosity
    pytest_args.append("-" + "v" * args.verbose)

    # Add capture setting
    pytest_args.append(f"--capture={args.capture}")

    # Add environment
    pytest_args.extend(["--env", args.env])

    # Add API timeout if specified
    if args.api_timeout:
        pytest_args.extend(["--api-timeout", str(args.api_timeout)])

    # Add tags/markers if specified
    if args.tags:
        pytest_args.extend(["-m", args.tags])

    # Add parallel execution if specified
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])
        pytest_args.append("--dist=worksteal")  # Better load balancing

    # Add Playwright-specific arguments
    pytest_args.extend([
        f"--browser-channel={args.browser_channel}",
        "--headed" if args.headed else "--headless"
    ])

    if args.slowmo:
        pytest_args.extend(["--slowmo", str(args.slowmo)])

    # Add HTML report generation
    if args.report:
        timestamp = Path(__file__).parent.name
        report_name = f"reports/test_report_{args.env}_{timestamp}.html"
        pytest_args.extend([
            "--html", report_name,
            "--self-contained-html",
            f"--html-report-title={args.report_title}"
        ])

    # Add JSON report for CI/CD integration
    pytest_args.extend([
        "--json-report",
        "--json-report-file=reports/test_results.json"
    ])

    # Add logging configuration
    pytest_args.extend([
        "--log-cli-level=INFO",
        "--log-cli-format=%(asctime)s [%(levelname)8s] %(name)s - %(message)s",
        "--log-cli-date-format=%Y-%m-%d %H:%M:%S"
    ])

    # Add test execution options
    pytest_args.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Ensure all markers are defined
        "--durations=10"  # Show 10 slowest tests
    ])

    # Add retry configuration
    pytest_args.extend([
        "--reruns=2",  # Retry failed tests 2 times
        "--reruns-delay=1"  # Wait 1 second between retries
    ])

    # Add any additional pytest arguments
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    # Print command for debugging
    print(f"Executing: pytest {' '.join(pytest_args)}\n")

    # Ensure reports directory exists
    Path("reports").mkdir(exist_ok=True)

    # Run pytest
    try:
        exit_code = pytest.main(pytest_args)
    except KeyboardInterrupt:
        print("\n❌ Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        sys.exit(1)

    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed successfully!")
        print("📊 Check the HTML report for detailed results")
    elif exit_code == 1:
        print("❌ Some tests failed")
        print("📊 Check the HTML report for failure details")
    elif exit_code == 2:
        print("❌ Test execution was interrupted")
    elif exit_code == 3:
        print("❌ Internal error occurred")
    elif exit_code == 4:
        print("❌ pytest usage error")
    elif exit_code == 5:
        print("⚠️  No tests were collected")
    else:
        print(f"❌ Tests failed with exit code: {exit_code}")
    
    print(f"📁 Reports saved to: reports/")
    print(f"{'='*60}\n")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
