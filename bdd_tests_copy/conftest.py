import logging
import os
import sys
import platform
import atexit
import json
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
from utils.request_response_tracker import RequestResponseTracker

# Global tracking
_request_response_tracker = RequestResponseTracker()
_test_start_time = datetime.now()


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
        "--retry-count",
        action="store",
        default="2",
        type=int,
        help="Number of retries for failed tests (default: 2)"
    )
    parser.addoption(
        "--log-level",
        action="store",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level from configuration"
    )


def pytest_configure(config):
    """Configure test environment before test run."""
    # Set environment
    env = config.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    
    # Setup logging (only once)
    log_level = config.getoption("--log-level") or os.getenv("LOG_LEVEL", "INFO")
    LoggerSetup.setup(log_level=log_level)
    
    logger = get_logger(__name__)
    logger.info(f"=" * 60)
    logger.info(f"Starting test run in {env.upper()} environment")
    logger.info(f"Retry count: {config.getoption('--retry-count')}")
    logger.info(f"=" * 60)
    
    # Configure pytest-html plugin
    config._metadata = {
        'Environment': env.upper(),
        'Platform': platform.system(),
        'Python': platform.python_version(),
        'Test Framework': 'pytest-bdd',
        'Retry Count': str(config.getoption('--retry-count'))
    }
    
    # Register cleanup
    atexit.register(cleanup_config)


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """Extend pytest-html report with request/response details."""
    pytest_html = item.config.pluginmanager.getplugin('html')
    outcome = yield
    report = outcome.get_result()
    
    # Add request/response data to the report
    if report.when == "call":
        # Get the test context if available
        test_id = item.nodeid.split("::")[-1]
        
        # Get request/response history for this test
        history = _request_response_tracker.get_test_history(test_id)
        
        if history:
            # Create detailed HTML for request/response
            extra = getattr(report, 'extra', [])
            
            # Add formatted request/response data
            html_content = _request_response_tracker.format_history_as_html(history)
            if html_content:
                extra.append(pytest_html.extras.html(html_content))
            
            # Add JSON data as collapsible section
            json_data = json.dumps(history, indent=2, default=str)
            extra.append(pytest_html.extras.json(json_data, name="API Request/Response Details"))
            
            report.extra = extra


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
def request_response_tracker() -> RequestResponseTracker:
    """Provide request/response tracker for the session."""
    return _request_response_tracker


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
        pytest.skip(f"Database initialization failed: {e}")
    finally:
        if 'manager' in locals():
            manager.close()
            logger.info("Database manager closed")


@pytest.fixture(scope="session")
def api_client(config: Settings, request_response_tracker: RequestResponseTracker) -> Generator[ChatbotAPIClient, None, None]:
    """Create API client for the session."""
    logger = get_logger(__name__)
    logger.info("Initializing API client")
    
    # Create API client with request/response tracking
    client = ChatbotAPIClient(config, request_response_tracker)
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
def test_context(request) -> Dict[str, Any]:
    """Provide clean context for each test."""
    test_id = request.node.nodeid.split("::")[-1]
    return {
        'test_id': test_id,
        'start_time': datetime.now(),
        'test_data': {},
        'results': {},
        'response': None,
        'conversation_id': None,
        'interaction_id': None,
        'correlation_id': None,
        'step_results': [],
        'request_history': []
    }


# Parametrization with CSV data including new test cases
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV data."""
    if "test_data_row" in metafunc.fixturenames:
        data_loader = DataLoader()
        logger = get_logger(__name__)
        
        try:
            # Load main complaint data
            complaint_data = []
            try:
                complaint_data = data_loader.load_csv("complaint_data.csv")
            except FileNotFoundError:
                logger.info("complaint_data.csv not found, trying extended test cases")
            
            # Load extended test cases
            extended_data = []
            try:
                extended_data = data_loader.load_csv("extended_test_cases.csv")
            except FileNotFoundError:
                logger.info("extended_test_cases.csv not found")
            
            # Combine all test data
            all_test_data = complaint_data + extended_data
            
            if all_test_data:
                test_ids = [
                    row.get("test_case_id", f"TC_{idx:03d}")
                    for idx, row in enumerate(all_test_data)
                ]
                
                logger.info(f"Loaded {len(all_test_data)} test cases")
                metafunc.parametrize("test_data_row", all_test_data, ids=test_ids)
            else:
                logger.warning("No test data found")
                metafunc.parametrize("test_data_row", [{}], ids=["NO_DATA"])
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            pytest.fail(f"Test data loading failed: {e}")
