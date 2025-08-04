import pytest
import os
import platform
from datetime import datetime
from typing import Dict, Any, List, Generator
import atexit
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import get_config, cleanup_config
from config.settings import Settings
from fixtures.api_service import ChatbotAPIClient
from fixtures.db import db_connection, db_utils
from utils.data_loader import DataLoader
from utils.logger_config import get_logger
from utils.report_generator import BusinessReportGenerator

# Global test results for reporting
_test_results: List[Dict[str, Any]] = []
_test_start_time = datetime.now()


def pytest_addoption(parser):
    """Add command-line options for runtime configuration."""
    parser.addoption(
        "--env", 
        action="store", 
        default="qa",
        choices=["development", "qa", "staging", "production"],
        help="Environment to run tests against"
    )
    parser.addoption(
        "--report", 
        action="store_true", 
        default=True,
        help="Generate HTML report after test run"
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure test environment before test run."""
    # Set environment
    os.environ["ENVIRONMENT"] = config.getoption("--env")
    
    logger = get_logger(__name__)
    logger.info(f"Starting test run in {os.environ['ENVIRONMENT']} environment")
    
    # Register cleanup
    atexit.register(cleanup_config)


def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    logger = get_logger(__name__)
    
    if _test_results and session.config.getoption("--report"):
        try:
            # Generate business report
            report_gen = BusinessReportGenerator()
            metadata = {
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'total_duration': (datetime.now() - _test_start_time).total_seconds(),
                'exit_status': exitstatus
            }
            
            report_path = report_gen.generate_report(_test_results, metadata)
            logger.info(f"Business report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


@pytest.fixture(scope="session")
def config() -> Settings:
    """Load configuration once per session."""
    return get_config()


@pytest.fixture(scope="session")
def api_client(config: Settings) -> Generator[ChatbotAPIClient, None, None]:
    """Create API client for the session."""
    client = ChatbotAPIClient(config)
    yield client
    client.close()


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
        'correlation_id': None
    }


# Test Result Collection
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect test results for business reporting."""
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "call":
        test_result = {
            'test_id': item.nodeid.split("::")[-1],
            'description': getattr(item, 'description', item.name),
            'status': report.outcome,
            'duration': report.duration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tags': [marker.name for marker in item.iter_markers()]
        }
        
        if hasattr(report, 'longrepr') and report.longrepr:
            test_result['error'] = str(report.longrepr)
        
        _test_results.append(test_result)


# BDD-specific fixtures
@pytest.fixture
def given_api_is_available(
    api_client: ChatbotAPIClient, 
    test_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Ensure API is available before test."""
    test_context['api_client'] = api_client
    return test_context


@pytest.fixture
def given_test_data_loaded(
    data_loader: DataLoader, 
    test_context: Dict[str, Any],
    test_data_row: Dict[str, Any]
) -> Dict[str, Any]:
    """Load test data for scenario."""
    test_context['test_data'] = test_data_row
    test_context['data_loader'] = data_loader
    return test_context


# Parametrization
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV data."""
    if "test_data_row" in metafunc.fixturenames:
        data_loader = DataLoader()
        try:
            test_data = data_loader.load_csv("complaint_data.csv")
            
            if test_data:
                test_ids = [
                    row.get("test_case_id", f"TC_{idx}") 
                    for idx, row in enumerate(test_data)
                ]
                metafunc.parametrize("test_data_row", test_data, ids=test_ids)
            else:
                # Provide empty data to avoid test collection errors
                metafunc.parametrize("test_data_row", [{}], ids=["NO_DATA"])
                
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to load test data: {e}")
            # Provide empty data to avoid test collection errors
            metafunc.parametrize("test_data_row", [{}], ids=["ERROR"])