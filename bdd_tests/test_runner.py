import sys
import os
import argparse
from pathlib import Path
import pytest
import logging

logger = logging.getLogger(__name__)

BASE_FEATURE_DIR = Path("bdd_tests").resolve()

def _is_within_base(p: Path, base: Path) -> bool:
    try:
        p = p.resolve(strict=False)
        base = base.resolve(strict=False)
        return base in p.parents or p == base
    except OSError:
        return False

def normalize_feature_path(feature: str) -> Path | None:
    
    candidate = Path(feature)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate)
    candidate = candidate.resolve(strict=False)
    if _is_within_base(candidate, BASE_FEATURE_DIR) and (candidate.is_dir() or candidate.suffix == ".feature"):
        return candidate

    candidate = (BASE_FEATURE_DIR / feature).resolve(strict=False)
    if _is_within_base(candidate, BASE_FEATURE_DIR) and (candidate.is_dir() or candidate.suffix == ".feature"):
        return candidate

    logger.error(f"Rejected path outside allowed directory: {feature}")
    return None

def main() -> int:
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="BDD Test Automation Framework Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="******",
    )
    parser.add_argument("--env", default="qa",
                        choices=["dev", "qa", "staging", "production"],
                        help="Environment to run tests against (default: qa)")
    parser.add_argument("--feature", nargs="*",
                        help="Specific feature file(s) or directory to run (e.g., features/login.feature or features/)")
    parser.add_argument("--tags", help="Run tests with specific tags (e.g., @smoke, @regression)")
    parser.add_argument("--report", action="store_true", help="Generate detailed HTML report")
    args = parser.parse_args()

    # Force base env vars early (no secrets)
    os.environ["ENVIRONMENT"] = args.env
    os.environ["LOG_LEVEL"] = "INFO"

    pytest_args: list[str] = []

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
        # default: whole features folder
        if not BASE_FEATURE_DIR.joinpath("features").exists():
            logger.error("Error: Features directory not found")
            return 1
        pytest_args.append(str(BASE_FEATURE_DIR / "features"))

    # env
    pytest_args.extend(["--env", args.env])

    # tags
    if args.tags:
        pytest_args.extend(["-m", args.tags])

    # report
    if args.report:
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        pytest_args.extend([f"--html={reports_dir}/report.html", "--self-contained-html"])

    # logging
    pytest_args.extend([
        "--log-cli-level=INFO",
        '--log-cli-format="%(asctime)s %(levelname)s %(name)s - %(message)s"',
        "--tb=short",
    ])

    logger.info("=" * 60)
    logger.info("Starting BDD Test Execution")
    logger.info(f"Environment: {args.env.upper()}")
    if args.feature:
        logger.info(f"Features: {', '.join(args.feature)}")
    if args.tags:
        logger.info(f"Tags: {args.tags}")
    logger.info("=" * 60)
    logger.info(f"Executing: pytest {' '.join(pytest_args)}")

    return pytest.main(pytest_args)

if __name__ == "__main__":
    sys.exit(main())
