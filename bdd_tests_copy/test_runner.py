import sys
import os
import argparse
import subprocess
import logging
from pathlib import Path
import pytest

def ensure_playwright_installed():
“”“Ensure Playwright is installed for API testing.”””
try:
import playwright
logger = logging.getLogger(**name**)
logger.info(“Playwright package found”)

```
    # For API-only testing, we don't need browsers, but check if command is available
    result = subprocess.run(
        ["python", "-m", "playwright", "--help"], 
        capture_output=True, 
        text=True,
        timeout=10
    )
    
    if result.returncode != 0:
        print("Installing Playwright CLI...")
        subprocess.run(["python", "-m", "pip", "install", "playwright"], check=True)
        print("Playwright CLI installed successfully.")
    else:
        print("Playwright CLI is available.")
        
    # Note: We skip browser installation for API-only testing
    print("ℹ️  Browsers skipped - API-only testing mode")
    
except ImportError:
    print("Error: Playwright package not found")
    print("Please run: pip install playwright")
    sys.exit(1)
except subprocess.CalledProcessError as e:
    print(f"Error with Playwright installation: {e}")
    print("Please run: pip install playwright")
    sys.exit(1)
except subprocess.TimeoutExpired:
    print("Warning: Playwright check timed out, continuing anyway...")
```

def _is_within_base(p: Path, base: Path) -> bool:
“”“Check if path is within the base directory.”””
try:
p = p.resolve(strict=False)
base = base.resolve(strict=False)
return base in p.parents or p == base
except OSError:
return False

def normalize_feature_path(feature: str) -> Path:
“”“Normalize and validate feature path for safety.”””
BASE_FEATURE_DIR = Path(“bdd_tests”).resolve()

```
candidate = Path(feature)
if not candidate.is_absolute():
    candidate = (Path.cwd() / candidate)

candidate = candidate.resolve(strict=False)
if _is_within_base(candidate, BASE_FEATURE_DIR) and (candidate.is_dir() or candidate.suffix == ".feature"):
    return candidate

candidate = (BASE_FEATURE_DIR / feature).resolve(strict=False)
if _is_within_base(candidate, BASE_FEATURE_DIR) and (candidate.is_dir() or candidate.suffix == ".feature"):
    return candidate

logging.getLogger(__name__).error(f"Rejected path outside allowed directory: {feature}")
return None
```

def main():
“”“Main entry point for Playwright API-based test runner.”””
parser = argparse.ArgumentParser(
description=“BDD Test Automation Framework Runner - Playwright API Edition”,
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog=”””
Examples:

# Run all tests in QA environment using Playwright API

python test_runner.py –env qa

# Run smoke tests in DEV environment with detailed report

python test_runner.py –env dev –tags smoke –report

# Run specific feature file with API timeout override

python test_runner.py –env qa –feature features/complaint_capture.feature –api-timeout 60

# Run tests in parallel with custom retry settings

python test_runner.py –env qa –parallel 4 –api-retry-count 5

# Run with enhanced Playwright API debugging

python test_runner.py –env qa –verbose -vv –playwright-debug
“””
)

```
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
    "--feature",
    help="Specific feature file or directory to run (e.g., features/complaint_capture.feature)"
)

parser.add_argument(
    "--report",
    action="store_true",
    default=True,
    help="Generate HTML report after test run (default: True)"
)

parser.add_argument(
    "--report-title",
    default="BDD Test Automation Report - Playwright API",
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
    "--api-retry-count",
    type=int,
    metavar="COUNT",
    help="Override API retry count"
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
    "--playwright-debug",
    action="store_true",
    help="Enable Playwright debugging (for API request troubleshooting)"
)

parser.add_argument(
    "--headed",
    action="store_true",
    help="This option is ignored for API-only testing"
)

parser.add_argument(
    "--slowmo",
    type=int,
    default=0,
    help="This option is ignored for API-only testing"
)

parser.add_argument(
    "--max-failures",
    type=int,
    metavar="N",
    help="Stop after N failures"
)

parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what tests would be run without executing them"
)

parser.add_argument(
    "pytest_args",
    nargs=argparse.REMAINDER,
    help="Additional arguments to pass to pytest"
)

args = parser.parse_args()

# Warn about ignored browser options
if args.headed:
    print("⚠️  --headed option ignored (API-only testing mode)")
if args.slowmo:
    print("⚠️  --slowmo option ignored (API-only testing mode)")

# Ensure Playwright is properly installed
ensure_playwright_installed()

# Set environment variable for configuration loading
os.environ["ENVIRONMENT"] = args.env
print(f"\n{'='*70}")
print(f"🚀 Starting Playwright API-based BDD Test Execution")
print(f"🌍 Environment: {args.env.upper()}")
print(f"🔧 Framework: pytest-bdd + Playwright APIRequestContext")
print(f"🎯 Mode: API-only testing (no browsers)")
print(f"{'='*70}\n")

# Build pytest command
pytest_args = []

# Resolve features safely
if args.feature:
    resolved: list[str] = []
    for f in args.feature:
        safe = normalize_feature_path(f)
        if not safe:
            return 1
        resolved.append(str(safe))
    pytest_args.extend(resolved)
else:
    # Default: whole features folder
    if not Path("bdd_tests").joinpath("features").exists():
        logging.getLogger(__name__).error("Error: Features directory not found")
        return 1
    pytest_args.append(str(Path("bdd_tests") / "features"))

# Add verbosity
pytest_args.append("-" + "v" * args.verbose)

# Add capture setting
pytest_args.append(f"--capture={args.capture}")

# Add environment
pytest_args.extend(["--env", args.env])

# Add API timeout if specified
if args.api_timeout:
    pytest_args.extend(["--api-timeout", str(args.api_timeout)])

# Add API retry count if specified
if args.api_retry_count:
    pytest_args.extend(["--api-retry-count", str(args.api_retry_count)])

# Add tags/markers if specified
if args.tags:
    pytest_args.extend(["-m", args.tags])

# Add parallel execution if specified
if args.parallel:
    pytest_args.extend(["-n", str(args.parallel)])
    pytest_args.append("--dist=worksteal")  # Better load balancing

# Add Playwright-specific arguments (API-only)
pytest_args.extend([
    "--headless",  # Always headless for API testing
    "--browser-channel=chrome"  # Dummy value (not used for API)
])

# Add Playwright debugging if requested
if args.playwright_debug:
    pytest_args.extend([
        "--log-cli-level=DEBUG",
        "--capture=no"
    ])
    os.environ["PWDEBUG"] = "console"

# Add HTML report generation
if args.report:
    timestamp = Path(__file__).parent.name
    report_name = f"reports/playwright_api_test_report_{args.env}_{timestamp}.html"
    pytest_args.extend([
        "--html", report_name,
        "--self-contained-html",
        f"--html-report-title={args.report_title}"
    ])

# Add JSON report for CI/CD integration
pytest_args.extend([
    "--json-report",
    "--json-report-file=reports/playwright_api_test_results.json"
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

# Add max failures if specified
if args.max_failures:
    pytest_args.extend(["--maxfail", str(args.max_failures)])

# Add dry run if specified
if args.dry_run:
    pytest_args.append("--collect-only")

# Add any additional pytest arguments
if args.pytest_args:
    pytest_args.extend(args.pytest_args)

# Print command for debugging
command_str = ' '.join(pytest_args)
print(f"🔧 Executing: pytest {command_str}\n")

# Ensure reports directory exists
Path("reports").mkdir(exist_ok=True)

# Set Playwright environment variables for API-only testing
os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"  # Skip browser downloads
os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"

# Run pytest
try:
    if args.dry_run:
        print("🔍 Dry run mode - showing what would be executed:")
        print("-" * 50)
    
    exit_code = pytest.main(pytest_args)
    
except KeyboardInterrupt:
    print("\n❌ Test execution interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Test execution failed with error: {e}")
    sys.exit(1)

# Print summary
print(f"\n{'='*70}")
if exit_code == 0:
    print("✅ All Playwright API tests passed successfully!")
    print("📊 Check the HTML report for detailed results")
elif exit_code == 1:
    print("❌ Some Playwright API tests failed")
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

print(f"🎯 API Framework: Playwright APIRequestContext")
print(f"🌍 Environment: {args.env.upper()}")
print(f"📁 Reports saved to: reports/")
print(f"{'='*70}\n")

# Additional debugging information
if exit_code != 0 and args.playwright_debug:
    print("🔍 Debugging Information:")
    print("- Check logs for detailed API request/response information")
    print("- Review the HTML report for step-by-step execution details")
    print("- Verify API endpoints and authentication in your .env files")

sys.exit(exit_code)
```

if **name** == “**main**”:
main()
