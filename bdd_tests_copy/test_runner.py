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
  # Run all tests in QA environment with 2 retries
  python test_runner.py --env qa --retry-count 2
  
  # Run smoke tests in DEV environment with HTML report
  python test_runner.py --env dev --tags smoke
  
  # Run specific test cases
  python test_runner.py --env qa --test-ids TC001,TC002,TC003
  
  # Run with parallel execution
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
        help="Pytest markers to run (e.g., 'smoke', 'regression', 'api and not slow')"
    )
    
    parser.add_argument(
        "--test-ids",
        help="Comma-separated list of test case IDs to run (e.g., TC001,TC002)"
    )
    
    parser.add_argument(
        "--retry-count",
        type=int,
        default=2,
        help="Number of times to retry failed tests (default: 2)"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=1,
        help="Delay in seconds between retries (default: 1)"
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
        "--report-dir",
        default="reports",
        help="Directory for test reports (default: reports)"
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
    print(f"BDD Test Automation Framework")
    print(f"Environment: {args.env.upper()}")
    print(f"Retry Count: {args.retry_count}")
    print(f"{'='*60}\n")
    
    # Ensure reports directory exists
    reports_dir = Path(args.report_dir)
    reports_dir.mkdir(exist_ok=True)
    
    # Build pytest command
    pytest_args = []
    
    # Add test directory
    pytest_args.append("bdd_tests/features/steps")
    
    # Add verbosity
    pytest_args.append("-" + "v" * args.verbose)
    
    # Add capture setting
    pytest_args.append(f"--capture={args.capture}")
    
    # Add environment
    pytest_args.extend(["--env", args.env])
    
    # Add retry configuration
    pytest_args.extend([
        f"--reruns={args.retry_count}",
        f"--reruns-delay={args.retry_delay}",
        f"--retry-count={args.retry_count}"  # Pass to our custom fixture
    ])
    
    # Add tags/markers if specified
    if args.tags:
        pytest_args.extend(["-m", args.tags])
    
    # Add specific test IDs if specified
    if args.test_ids:
        test_ids = args.test_ids.split(',')
        for test_id in test_ids:
            pytest_args.extend(["-k", test_id])
    
    # Add parallel execution if specified
    if args.parallel:
        pytest_args.extend(["-n", str(args.parallel)])
    
    # HTML report configuration
    report_path = reports_dir / f"test_report_{args.env}_{Path(__file__).parent.name}.html"
    pytest_args.extend([
        f"--html={report_path}",
        "--self-contained-html",
        "--html-report-title=BDD Test Automation Report"
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
        print(f"❌ Tests completed with exit code: {exit_code}")
    
    print(f"📊 Report generated: {report_path}")
    print(f"{'='*60}\n")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
