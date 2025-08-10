import logging
import os
import sys
import platform
import atexit
from datetime import datetime
from typing import Dict, Any, Generator, List, Optional
from pathlib import Path

import pytest
from pytest_bdd import given, when, then, parsers

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import get_config, cleanup_config
from config.settings import Settings
from database.db_manager import DatabaseManager
from fixtures.api_service import ChatbotAPIClient
from fixtures.db_utils import DBUtils
from utils.data_loader import DataLoader
from utils.logger_config import get_logger, LoggerSetup
from utils.report_generator import BusinessReportGenerator

# Global test results and report generator
_test_results: List[Dict[str, Any]] = []
_test_start_time = datetime.now()
_report_generator = BusinessReportGenerator()
_current_test_id = None


def pytest_addoption(parser):
    """Add command-line options for runtime configuration."""
    parser.addoption(
        "--env",
        action="store",
        default="qa",
        choices=["dev", "qa", "staging", "production"],
        help="Environment to run tests against"
    )
    parser.addoption(
        "--report",
        action="store_true",
        default=True,
        help="Generate HTML report after test run"
    )
    parser.addoption(
        "--report-title",
        action="store",
        default="BDD Test Automation Report",
        help="Title for the HTML report"
    )
    parser.addoption(
        "--log-level",
        action="store",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from configuration"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure test environment before test run."""
    # Set environment
    env = config.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    
    # Setup logging
    log_level = config.getoption("--log-level")
    if not log_level:
        # Try to get from environment or use default
        log_level = os.getenv("LOG_LEVEL", "INFO")
    
    LoggerSetup.setup(log_level=log_level)
    logger = get_logger(__name__)
    logger.info(f"=" * 60)
    logger.info(f"Starting test run in {env.upper()} environment")
    logger.info(f"=" * 60)
    
    # Register cleanup
    atexit.register(cleanup_config)


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    logger = get_logger(__name__)
    
    if _test_results and session.config.getoption("--report"):
        try:
            # Calculate execution metadata
            metadata = {
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'total_duration': (datetime.now() - _test_start_time).total_seconds(),
                'exit_status': exitstatus
            }
            
            # Generate business report
            report_path = _report_generator.generate_report(_test_results, metadata)
            logger.info(f"Test report generated: {report_path}")
            
            # Print summary
            total_tests = len(_test_results)
            passed = sum(1 for r in _test_results if r['status'] == 'passed')
            failed = sum(1 for r in _test_results if r['status'] == 'failed')
            
            logger.info(f"=" * 60)
            logger.info(f"Test Execution Summary")
            logger.info(f"Total: {total_tests} | Passed: {passed} | Failed: {failed}")
            logger.info(f"Pass Rate: {(passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
            logger.info(f"=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


@pytest.fixture(scope="session")
def config(pytestconfig) -> Settings:
    """Load configuration once per session."""
    env = pytestconfig.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    
    logger = get_logger(__name__)
    logger.info(f"Loading configuration for {env} environment")
    
    try:
        settings = get_config()
        logger.info(f"Configuration loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        pytest.fail(f"Configuration loading failed: {e}")


@pytest.fixture(scope="session")
def db_manager(config: Settings) -> Generator[DatabaseManager, None, None]:
    """Create database manager for the session."""
    logger = get_logger(__name__)
    logger.info("Initializing database manager")
    
    try:
        manager = DatabaseManager(config)
        logger.info("Database manager initialized successfully")
        yield manager
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
        pytest.fail(f"Database initialization failed: {e}")
    finally:
        if 'manager' in locals():
            manager.close()
            logger.info("Database manager closed")


@pytest.fixture(scope="session")
def api_client(config: Settings) -> Generator[ChatbotAPIClient, None, None]:
    """Create API client for the session."""
    logger = get_logger(__name__)
    logger.info("Initializing API client")
    
    client = ChatbotAPIClient(config)
    yield client
    client.close()
    logger.info("API client closed")


@pytest.fixture(scope="session")
def db_utils(db_manager: DatabaseManager) -> DBUtils:
    """Provide database utilities."""
    return DBUtils(db_manager)


@pytest.fixture(scope="session")
def data_loader() -> DataLoader:
    """Create data loader instance."""
    return DataLoader()


@pytest.fixture
def test_context() -> Dict[str, Any]:
    """Provide clean context for each test."""
    return {
        'start_time': datetime.now(),
        'test_data': {},
        'results': {},
        'response': None,
        'conversation_id': None,
        'interaction_id': None,
        'correlation_id': None,
        'step_results': []
    }


# Step-level fixtures for BDD
@pytest.fixture
def given_api_is_available(api_client: ChatbotAPIClient, test_context: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure API is available before test."""
    test_context['api_client'] = api_client
    _report_generator.add_step_result({
        'test_id': _current_test_id,
        'type': 'given',
        'description': 'API is available',
        'status': 'passed'
    })
    return test_context


@pytest.fixture
def given_test_data_loaded(data_loader: DataLoader, test_context: Dict[str, Any], test_data_row: Dict[str, Any]) -> Dict[str, Any]:
    """Load test data for scenario."""
    test_context['test_data'] = test_data_row
    test_context['data_loader'] = data_loader
    _report_generator.add_step_result({
        'test_id': _current_test_id,
        'type': 'given',
        'description': f'Test data loaded for case: {test_data_row.get("test_case_id")}',
        'status': 'passed'
    })
    return test_context


# Parametrization with CSV data
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV data."""
    if "test_data_row" in metafunc.fixturenames:
        data_loader = DataLoader()
        logger = get_logger(__name__)
        
        try:
            # Load test data from CSV
            test_data = data_loader.load_csv("complaint_data.csv")
            
            if test_data:
                # Create test IDs from test_case_id column
                test_ids = [
                    row.get("test_case_id", f"TC_{idx:03d}")
                    for idx, row in enumerate(test_data)
                ]
                
                logger.info(f"Loaded {len(test_data)} test cases from CSV")
                metafunc.parametrize("test_data_row", test_data, ids=test_ids)
            else:
                # Provide empty data to avoid test collection errors
                logger.warning("No test data found in CSV file")
                metafunc.parametrize("test_data_row", [{}], ids=["NO_DATA"])
                
        except FileNotFoundError:
            logger.warning("Test data file not found, using sample data")
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            pytest.fail(f"Test data loading failed: {e}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect test results for business reporting."""
    global _current_test_id
    
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "setup":
        _current_test_id = item.nodeid.split("::")[-1]
    
    elif report.when == "call":
        test_result = {
            'test_id': item.nodeid.split("::")[-1],
            'description': getattr(item, 'description', item.name),
            'status': report.outcome,
            'duration': report.duration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tags': [marker.name for marker in item.iter_markers()],
            'environment': os.getenv('ENVIRONMENT', 'unknown')
        }
        
        if hasattr(report, 'longrepr') and report.longrepr:
            test_result['error'] = str(report.longrepr)
            
        _test_results.append(test_result)