import pytest
import os
import platform
from datetime import datetime
from typing import Dict, Any, List
import atexit

from config.config_loader import get_config, cleanup_config
from config.settings import Settings
from core.api.chatbot_client import ChatbotAPIClient
from core.database.db_manager import DatabaseManager
from core.database.db_operations import DatabaseOperations
from core.utils.data_loader import DataLoader
from core.utils.logger import get_logger, LoggerSetup
from reporting.report_generator import BusinessReportGenerator

# Global test results for reporting
_test_results: List[Dict[str, Any]] = []
_test_start_time = datetime.now()

def pytest_addoption(parser):
    """Add command-line options for runtime configuration."""
    parser.addoption("--env", action="store", default="qa",
    help="Environment to run tests against (development/qa/staging/production)")

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configure test environment before test run."""
    # Set environment
    os.environ["ENVIRONMENT"] = config.getoption("--env")

    # Setup logging
    LoggerSetup.setup(log_level=os.getenv("LOG_LEVEL", "INFO"))

    logger = get_logger(__name__)
    logger.info(f"Starting test run in {os.environ['ENVIRONMENT']} environment")

# Register cleanup
atexit.register(cleanup_config)
def pytest_sessionfinish(session, exitstatus): 
    """Generate report after all tests complete.""" logger = get_logger(name)
    if _test_results:
                try:
                    # Generate business report
                    report_gen = BusinessReportGenerator()
                    metadata = {
                        'environment': os.getenv('ENVIRONMENT', 'unknown'),
                        'platform': platform.system(),
                        'python_version': platform.python_version(),
                        'total_duration': (datetime.now() - _test_start_time).total_seconds()
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
def api_client(config: Settings) -> ChatbotAPIClient:
    """Create API client for the session."""
    client = ChatbotAPIClient(config)
    yield client
    client.close()

@pytest.fixture(scope="session")
def db_manager(config: Settings) -> DatabaseManager:
    """Create database manager for the session."""
    manager = DatabaseManager(config)
    yield manager
    manager.close()

@pytest.fixture(scope="session")
def db_operations(db_manager: DatabaseManager) -> DatabaseOperations:
    """Create database operations instance."""
    return DatabaseOperations(db_manager)

@pytest.fixture(scope="session")
def data_loader() -> DataLoader:
    """Create data loader instance."""
    return DataLoader()

@pytest.fixture(scope="function")
def test_context() -> Dict[str, Any]:
    """Provide clean context for each test."""
    return {
    'start_time': datetime.now(),
    'test_data': {},
    'results': {}
    }

# --- Test Result Collection ---
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Collect test results for business reporting."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call":
        test_result = {
            'test_id': item.nodeid.split("::")[-1],
            'description': item.name,
            'status': report.outcome,
            'duration': report.duration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if hasattr(report, 'longrepr') and report.longrepr:
            test_result['error'] = str(report.longrepr)
        
        _test_results.append(test_result)
        
#--- BDD-specific fixtures ---
@pytest.fixture
def given_api_is_available(api_client: ChatbotAPIClient, test_context: Dict[str, Any]):
    """Ensure API is available before test."""
    test_context['api_client'] = api_client
    return test_context

@pytest.fixture
def given_test_data_loaded(data_loader: DataLoader, test_context: Dict[str, Any],
    test_data_row: Dict[str, Any]):
    """Load test data for scenario."""
    test_context['test_data'] = test_data_row
    test_context['data_loader'] = data_loader
    return test_context

#--- Parametrization ---
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV data."""
    if "test_data_row" in metafunc.fixturenames:
        data_loader = DataLoader()
        test_data = data_loader.load_csv("complaint_data.csv")

    if test_data:
            test_ids = [row.get("test_case_id", f"TC_{idx}") for idx, row in enumerate(test_data)]
            metafunc.parametrize("test_data_row", test_data, ids=test_ids)
