import pytest
import argparse
import os
import sys

def main():

    parser = argparse.ArgumentParser(description="Test Runner for BDD Framework")
    parser.add_argument(
        "--env",
        default="qa",
        help="Environment to run tests against (e.g., 'qa', 'dev'). This will select the .env file."
    )
    parser.add_argument(
        "--tags",
        help="Pytest markers to run (e.g., 'smoke and not regression')."
    )
    parser.add_argument(
        "--feature",
        help="Run a specific feature file or directory."
    )
    parser.add_argument(
        "--report-title",
        default="Test Automation Report",
        help="Custom title for the HTML report."
    )
    parser.add_argument(
        '--pytest-args', 
        nargs=argparse.REMAINDER,
        help="Any other arguments to pass directly to pytest."
    )

    args = parser.parse_args()

    # --- Construct the pytest command ---
    pytest_command = [
        # Add the root test directory
        "bdd_tests/",
        # Make test output verbose
        "-v",
        # Capture logs for the report
        "--log-cli-level=INFO",
        # Pass the environment to our custom hook in conftest.py
        f"--env={args.env}",
        # Enable HTML reporting
        "--html-report",
        # Pass the custom report title
        f"--html-report-title={args.report_title}"
    ]

    # Add tags if provided
    if args.tags:
        pytest_command.append(f"-m {args.tags}")

    # Add specific feature file path if provided
    if args.feature:
        pytest_command[0] = args.feature

    # Add any extra pytest arguments
    if args.pytest_args:
        pytest_command.extend(args.pytest_args)
        
    print(f"Running pytest with command: {' '.join(pytest_command)}")

    # --- Execute pytest ---
    # Using pytest.main is cleaner than subprocess and gives a proper exit code.
    exit_code = pytest.main(pytest_command)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
