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
from playwright.sync_api import sync_playwright, Playwright, APIRequestContext

# Add project root to path

sys.path.insert(0, os.path.dirname(os.path.abspath(**file**)))

from config.loader import get_config, cleanup_config
from config.settings import Settings
from database.db_manager import DatabaseManager
from fixtures.api_service import ChatbotAPIClient, PlaywrightAPIHelpers
from fixtures.db_utils import DBUtils
from utils.data_loader import DataLoader
from utils.logger_config import get_logger, LoggerSetup
from utils.request_response_tracker import RequestResponseTracker

# Global tracking

_request_response_tracker = RequestResponseTracker()
_test_start_time = datetime.now()

def pytest_addoption(parser):
“”“Add command-line options for runtime configuration.”””
parser.addoption(
“–env”,
action=“store”,
default=“qa”,
choices=[“dev”, “qa”, “staging”, “production”],
help=“Environment to run tests against”
)
parser.addoption(
“–retry-count”,
action=“store”,
default=“2”,
type=int,
help=“Number of retries for failed tests (default: 2)”
)
parser.addoption(
“–api-timeout”,
action=“store”,
default=None,
type=int,
help=“Override API timeout in seconds”
)
parser.addoption(
“–api-retry-count”,
action=“store”,
default=None,
type=int,
help=“Override API retry count”
)
parser.addoption(
“–playwright-headless”,
action=“store_true”,
default=True,
help=“Run Playwright in headless mode (API-only testing)”
)

def pytest_configure(config):
“”“Configure test environment before test run.”””
# Set environment
env = config.getoption(”–env”)
os.environ[“ENVIRONMENT”] = env

```
# Setup logging (only once)
log_level = config.getoption("--log-level") or os.getenv("LOG_LEVEL", "INFO")
LoggerSetup.setup(log_level=log_level)

logger = get_logger(__name__)
logger.info(f"=" * 60)
logger.info(f"Starting Playwright API-only test run in {env.upper()} environment")
logger.info(f"Retry count: {config.getoption('--retry-count')}")
logger.info(f"API timeout override: {config.getoption('--api-timeout')}")
logger.info(f"API retry override: {config.getoption('--api-retry-count')}")
logger.info(f"=" * 60)

# Configure pytest-html plugin metadata
config._metadata = {
    'Environment': env.upper(),
    'Platform': platform.system(),
    'Python': platform.python_version(),
    'Test Framework': 'pytest-bdd + Playwright API',
    'Retry Count': str(config.getoption('--retry-count')),
    'API Testing': 'Playwright APIRequestContext Only',
    'Browser Mode': 'API-only (No Browser)'
}

# Register cleanup
atexit.register(cleanup_config)
```

def pytest_configure_playwright(config):
“”“Configure Playwright settings for API-only testing.”””
logger = get_logger(**name**)

```
try:
    # Set Playwright-specific environment variables for API-only testing
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")  # Skip browser downloads
    
    # Configure Playwright for API-only usage
    playwright_config = {
        # API Request specific settings
        "timeout": int(os.getenv("API_TIMEOUT", "45")) * 1000,  # Convert to milliseconds
        "ignore_https_errors": not bool(os.getenv("VERIFY_SSL", "true").lower() == "true"),
        
        # Disable all browser-related features
        "trace": {
            "screenshots": False,  # Not needed for API testing
            "snapshots": False,    # Not needed for API testing
            "sources": True,       # Keep source code traces for debugging
        },
        
        # Disable video and screenshots (API-only)
        "video": None,
        "screenshot": None,
        
        # Global timeout settings for API operations
        "expect_timeout": 30000,  # 30 seconds for assertions
        "action_timeout": 60000,  # 60 seconds for actions
    }
    
    # Store configuration for later use
    config._playwright_api_config = playwright_config
    
    logger.info("Playwright configured for API-only testing")
    logger.debug(f"Playwright config: {playwright_config}")
    
except Exception as e:
    logger.warning(f"Failed to configure Playwright settings: {e}")
    # Don't fail the test run if Playwright configuration fails
    pass
```

@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
“”“Extend pytest-html report with Playwright API request/response details.”””
pytest_html = item.config.pluginmanager.getplugin(‘html’)
outcome = yield
report = outcome.get_result()

```
# Add request/response data to the report
if report.when == "call":
    # Get the test context if available
    test_id = item.nodeid.split("::")[-1]
    
    # Get request/response history for this test
    history = _request_response_tracker.get_test_history(test_id)
    
    if history:
        # Create detailed HTML for request/response
        extra = getattr(report, 'extra', [])
        
        # Add formatted request/response data with Playwright context
        html_content = _request_response_tracker.format_history_as_html(history)
        if html_content:
            # Add Playwright-specific styling
            playwright_html = f"""
            <div class="playwright-api-details">
                <h4>Playwright API Request/Response Details</h4>
                <div class="api-framework-info">
                    <span class="framework-badge">Playwright APIRequestContext</span>
                    <span class="mode-badge">API-Only Testing</span>
                </div>
                {html_content}
            </div>
            <style>
            .playwright-api-details {{ margin: 10px 0; }}
            .api-framework-info {{ margin-bottom: 10px; }}
            .framework-badge, .mode-badge {{ 
                background: #007acc; 
                color: white; 
                padding: 3px 8px; 
                border-radius: 3px; 
                font-size: 0.8em;
                margin-right: 5px;
            }}
            .mode-badge {{ background: #28a745; }}
            </style>
            """
            extra.append(pytest_html.extras.html(playwright_html))
        
        # Add JSON data as collapsible section
        json_data = json.dumps(history, indent=2, default=str)
        extra.append(pytest_html.extras.json(json_data, name="Playwright API Request/Response Details"))
        
        report.extra = extra
```

@pytest.fixture(scope=“session”)
def config(pytestconfig) -> Settings:
“”“Load configuration once per session with Playwright overrides.”””
env = pytestconfig.getoption(”–env”)
os.environ[“ENVIRONMENT”] = env

```
logger = get_logger(__name__)
logger.info(f"Loading configuration for {env} environment")

try:
    settings = get_config()
    
    # Override API timeout if specified
    api_timeout = pytestconfig.getoption("--api-timeout")
    if api_timeout:
        settings.API_TIMEOUT = api_timeout
        logger.info(f"API timeout overridden to {api_timeout}s")
    
    # Override API retry count if specified
    api_retry_count = pytestconfig.getoption("--api-retry-count")
    if api_retry_count:
        settings.API_RETRY_COUNT = api_retry_count
        logger.info(f"API retry count overridden to {api_retry_count}")
    
    logger.info(f"Configuration loaded successfully for Playwright API testing")
    return settings
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    pytest.fail(f"Configuration loading failed: {e}")
```

@pytest.fixture(scope=“session”)
def playwright() -> Generator[Playwright, None, None]:
“”“Create Playwright instance for API-only testing.”””
logger = get_logger(**name**)
logger.info(“Initializing Playwright for API-only testing (no browsers)”)

```
with sync_playwright() as p:
    # Ensure we're not starting any browsers
    logger.info("Playwright instance created for API testing only")
    yield p
```

@pytest.fixture(scope=“session”)
def playwright_request_context(
config: Settings,
playwright: Playwright
) -> Generator[APIRequestContext, None, None]:
“”“Create Playwright APIRequestContext for API-only testing.”””
logger = get_logger(**name**)
logger.info(“Creating Playwright API request context”)

```
try:
    # Configure SSL certificate if available
    client_certificates = []
    if config.CERT_PEM_PATH and config.KEY_PEM_PATH:
        client_certificates = [{
            "cert": config.CERT_PEM_PATH,
            "key": config.KEY_PEM_PATH
        }]
        logger.info("Client certificate configured for API requests")
    
    # Create request context with enhanced configuration for API testing
    context_options = {
        "base_url": config.API_BASE_URL,
        "timeout": config.API_TIMEOUT * 1000,  # Convert to milliseconds
        "ignore_https_errors": not config.VERIFY_SSL,
        "extra_http_headers": {
            'User-Agent': 'ChatbotAutomation-Playwright-API/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    }
    
    # Add client certificates if available
    if client_certificates:
        context_options["client_certificates"] = client_certificates
    
    context = playwright.request.new_context(**context_options)
    
    logger.info("Playwright API context initialized successfully")
    logger.info(f"Base URL: {config.API_BASE_URL}")
    logger.info(f"Timeout: {config.API_TIMEOUT}s")
    logger.info(f"SSL Verification: {config.VERIFY_SSL}")
    
    yield context
    
except Exception as e:
    logger.error(f"Failed to initialize Playwright API context: {e}")
    pytest.skip(f"Playwright API context initialization failed: {e}")
finally:
    if 'context' in locals():
        context.dispose()
        logger.info("Playwright API context disposed")
```

@pytest.fixture(scope=“session”)
def request_response_tracker() -> RequestResponseTracker:
“”“Provide request/response tracker for the session.”””
return _request_response_tracker

@pytest.fixture(scope=“session”)
def db_manager(config: Settings) -> Generator[DatabaseManager, None, None]:
“”“Create database manager for the session.”””
logger = get_logger(**name**)
logger.info(“Initializing database manager”)

```
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
```

@pytest.fixture(scope=“session”)
def api_client(
config: Settings,
request_response_tracker: RequestResponseTracker,
playwright_request_context: APIRequestContext
) -> Generator[ChatbotAPIClient, None, None]:
“”“Create Playwright-based API client for the session.”””
logger = get_logger(**name**)
logger.info(“Initializing Playwright API client”)

```
# Create API client with Playwright request context and tracking
client = ChatbotAPIClient(config, request_response_tracker, playwright_request_context)

try:
    # Perform health check if endpoint is available
    try:
        health_response = client.health_check()
        logger.info(f"API health check successful: {health_response}")
    except Exception as e:
        logger.warning(f"API health check failed (continuing anyway): {e}")
    
    yield client
    
except Exception as e:
    logger.error(f"Playwright API client initialization failed: {e}")
    pytest.skip(f"API client initialization failed: {e}")
finally:
    client.close()
    logger.info("Playwright API client closed")
```

@pytest.fixture(scope=“session”)
def playwright_helpers() -> PlaywrightAPIHelpers:
“”“Provide Playwright API helper utilities.”””
return PlaywrightAPIHelpers()

@pytest.fixture(scope=“session”)
def db_utils(db_manager: DatabaseManager) -> DBUtils:
“”“Provide database utilities.”””
return DBUtils(db_manager)

@pytest.fixture(scope=“session”)
def data_loader() -> DataLoader:
“”“Create data loader instance.”””
return DataLoader()

@pytest.fixture
def test_context(request) -> Dict[str, Any]:
“”“Provide clean context for each test with Playwright-specific features.”””
test_id = request.node.nodeid.split(”::”)[-1]
return {
‘test_id’: test_id,
‘start_time’: datetime.now(),
‘test_data’: {},
‘results’: {},
‘response’: None,
‘conversation_id’: None,
‘interaction_id’: None,
‘correlation_id’: None,
‘step_results’: [],
‘request_history’: [],
‘playwright_context’: ‘api_only’,  # Indicate this is API-only testing
‘api_framework’: ‘playwright’,
‘wait_strategies_used’: [],
‘performance_metrics’: {}
}

# Enhanced fixtures for step definitions compatibility with Playwright

@pytest.fixture
def given_api_is_available(
api_client: ChatbotAPIClient,
test_context: Dict[str, Any],
playwright_helpers: PlaywrightAPIHelpers
) -> Dict[str, Any]:
“”“Verify Playwright API is available and add to test context.”””
test_context[‘api_client’] = api_client
test_context[‘playwright_helpers’] = playwright_helpers

```
logger = get_logger(__name__)
logger.info(f"Playwright API client available for test: {test_context['test_id']}")

return test_context
```

@pytest.fixture
def given_test_data_loaded(test_data_row: Dict[str, Any], test_context: Dict[str, Any]) -> Dict[str, Any]:
“”“Load test data into context.”””
test_context[‘test_data’] = test_data_row

```
logger = get_logger(__name__)
test_case_id = test_data_row.get('test_case_id', 'Unknown')
logger.info(f"Test data loaded for Playwright API test case: {test_case_id}")

return test_context
```

def pytest_generate_tests(metafunc):
“”“Generate parameterized tests from CSV data.”””
if “test_data_row” in metafunc.fixturenames:
data_loader = DataLoader()
logger = get_logger(**name**)

```
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
            
            logger.info(f"Loaded {len(all_test_data)} test cases for Playwright API execution")
            metafunc.parametrize("test_data_row", all_test_data, ids=test_ids)
        else:
            logger.warning("No test data found")
            metafunc.parametrize("test_data_row", [{}], ids=["NO_DATA"])
            
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        pytest.fail(f"Test data loading failed: {e}")
```

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
“”“Setup individual test runs with Playwright context.”””
logger = get_logger(**name**)
test_name = item.nodeid.split(”::”)[-1]
logger.info(f”Setting up Playwright API test: {test_name}”)

```
# Set current test in tracker
if hasattr(item.session, '_request_response_tracker'):
    item.session._request_response_tracker.set_current_test(test_name)
```

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
“”“Teardown individual test runs.”””
logger = get_logger(**name**)
test_name = item.nodeid.split(”::”)[-1]
logger.info(f”Tearing down Playwright API test: {test_name}”)

def pytest_sessionfinish(session, exitstatus):
“”“Generate final report after all tests complete.”””
logger = get_logger(**name**)

```
try:
    # Calculate execution metadata
    metadata = {
        'environment': os.getenv('ENVIRONMENT', 'unknown'),
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'total_duration': (datetime.now() - _test_start_time).total_seconds(),
        'exit_status': exitstatus,
        'api_framework': 'playwright',
        'testing_mode': 'api_only'
    }
    
    logger.info(f"=" * 60)
    logger.info(f"Playwright API Test Execution Summary")
    logger.info(f"Total Duration: {metadata['total_duration']:.2f}s")
    logger.info(f"Exit Status: {exitstatus}")
    logger.info(f"API Framework: Playwright APIRequestContext")
    logger.info(f"Testing Mode: API-only (No Browser)")
    logger.info(f"=" * 60)
    
except Exception as e:
    logger.error(f"Failed to generate final report: {e}")
```

# Pytest plugin configuration for Playwright

def pytest_playwright_auto_setup(config):
“”“Auto-setup for Playwright when plugin is available.”””
logger = get_logger(**name**)
logger.info(“Playwright auto-setup for API-only testing framework”)

```
# Ensure we're in API-only mode
if hasattr(config, '_playwright_api_config'):
    logger.info("Playwright configured for API-only testing mode")
else:
    logger.warning("Playwright API configuration not found")
```
