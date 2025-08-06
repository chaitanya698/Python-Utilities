import pytest
import os
import logging
from datetime import datetime
from bdd_tests.config.loader import load_settings
from bdd_tests.database.db_manager import DatabaseManager
from bdd_tests.fixtures.api_service import ApiService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("conftest")

# --- Pytest Hooks ---

def pytest_addoption(parser):
    """Adds custom command-line options to pytest."""
    parser.addoption("--env", action="store", default="qa", help="Environment to run tests against (e.g., 'qa', 'dev')")
    parser.addoption("--html-report-title", action="store", default="Test Automation Report", help="Set the title of the HTML report")

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """
    Called after command line options have been parsed.
    This is the ideal place to load settings and initialize shared resources.
    """
    # 1. Load Environment Settings
    env = config.getoption("--env")
    logger.info(f"pytest_configure hook: Loading settings for environment '{env}'")
    load_settings(env)

    # 2. Initialize Database Pool
    # We wrap this in a try-except block to prevent the test run from starting
    # if the database is not accessible.
    try:
        logger.info("pytest_configure hook: Initializing database pool.")
        DatabaseManager.initialize_pool()
    except Exception as e:
        pytest.exit(f"Failed to connect to the database, aborting test run. Error: {e}", 1)

    # 3. Configure HTML Report
    report_title = config.getoption("--html-report-title")
    config.option.htmlpath = f"reports/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    config.option.css = ['reports/style.css']
    config.option.self_contained_html = True
    config.stash['report_title'] = report_title
    logger.info(f"HTML report will be saved to: {config.option.htmlpath}")

def pytest_unconfigure(config):
    """
    Called before test process is exited.
    Perfect place for cleanup activities.
    """
    logger.info("pytest_unconfigure hook: Closing database pool.")
    DatabaseManager.close_pool()

# --- Fixtures ---

@pytest.fixture(scope="session")
def api_service():
    """
    Session-scoped fixture to provide an instance of the ApiService.
    This ensures we use the same service object for all tests in a session.
    """
    logger.info("Creating ApiService instance for the test session.")
    return ApiService()

@pytest.fixture(scope="function")
def db_connection():
    """
    Function-scoped fixture to provide a database connection for a single test.
    It handles acquiring the connection from the pool and releasing it afterward.
    """
    logger.info("Acquiring database connection for a test.")
    connection = DatabaseManager.get_connection()
    try:
        # 'yield' passes the connection object to the test function.
        yield connection
    finally:
        # This code runs after the test function completes.
        logger.info("Releasing database connection back to the pool.")
        if connection:
            connection.close() # For oracledb, close() on a pooled connection returns it to the pool.

# --- HTML Report Customization ---

def pytest_html_report_title(report):
    """Sets the title of the HTML report."""
    report.title = report.config.stash.get('report_title', 'Default Test Report')

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """

    Hook to access the test report object and add extra information.
    We use this to add logs to the HTML report.
    """
    outcome = yield
    report = outcome.get_result()
    extra = getattr(report, "extra", [])
    if report.when == "call":
        # Check if there is captured log content and add it to the report
        if report.longreprtext:
            extra.append(pytest_html.extras.text(report.longreprtext, "Logs"))
        # You can add more details here, like screenshots on failure
        # if 'xfail' in report.keywords:
        #     extra.append(pytest_html.extras.url('http://www.example.com/'))
        report.extra = extra
