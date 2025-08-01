# bdd_tests/conftest.py

import pytest
import pytest_html
import os
import logging
from typing import Dict, Any

def pytest_addoption(parser):
    """Adds custom command-line options to pytest."""
    parser.addoption(
        "--env", action="store", default="qa", help="Environment config file to use (e.g., dev, qa)"
    )
    parser.addoption("--db-host", action="store", help="Database host IP or DNS name")
    parser.addoption("--db-port", action="store", help="Database port number")
    parser.addoption("--db-user", action="store", help="Database username")
    parser.addoption("--db-password", action="store", help="Database password")
    parser.addoption("--db-service-name", action="store", help="Oracle database service name")

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """
    This hook runs after command-line options are parsed.
    It sets environment variables from the provided options, which will override .env file values.
    """
    os.environ["TEST_ENV"] = config.getoption("--env")

    db_options = {
        "DB_HOST": config.getoption("--db-host"),
        "DB_PORT": config.getoption("--db-port"),
        "DB_USER": config.getoption("--db-user"),
        "DB_PASSWORD": config.getoption("--db-password"),
        "DB_SERVICE_NAME": config.getoption("--db-service-name"),
    }

    for key, value in db_options.items():
        if value is not None:
            os.environ[key] = str(value)
            print(f"✅ Overriding setting from command line: {key}")

    from bdd_tests.utils import logger_config

from bdd_tests.utils.api_service import ChatbotAPIService
from bdd_tests.utils.db_utils import DBUtils
from bdd_tests.config.db_config import engine

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def api_service() -> ChatbotAPIService:
    """Provides a single, session-scoped instance of the ChatbotAPIService."""
    return ChatbotAPIService()

@pytest.fixture(scope="session")
def db_utils() -> DBUtils:
    """
    Provides a session-scoped DB utility instance and handles engine teardown.
    """
    yield DBUtils()
    logger.info("Disposing of shared database engine connection pool at end of session.")
    engine.dispose()

@pytest.fixture(scope="function")
def chatbot_context() -> Dict[str, Any]:
    """Provides a clean dictionary for sharing state within a single scenario."""
    return {}

def pytest_html_report_title(report):
    """Sets a custom title for the HTML report."""
    report.title = "Chatbot Complaint AI - BDD Test Report"

def pytest_runtest_makereport(item, call):
    """Adds detailed BDD steps with status icons to the HTML report."""
    outcome = yield
    report = outcome.get_result()
    if report.when == "call" and hasattr(item, "bdd_step_results"):
        steps_html = "".join(
            f'<div><span class="status {status.lower()}">{icon}</span> {step_name}</div>'
            for step_name, status, icon in item.bdd_step_results
        )
        report.extra.append(pytest_html.extras.html(f"<h4>Scenario Steps:</h4>{steps_html}"))

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_scenario(request, feature, scenario):
    """Initializes a list to hold step results before each scenario."""
    request.node.bdd_step_results = []

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_after_step(request, step):
    """Records the result of each step right after it runs."""
    item = request.node
    status, icon = ("Passed", "✅")
    for rep in item.session.reports:
        if rep.when == "call" and rep.nodeid == item.nodeid and not rep.passed:
            status, icon = ("Failed", "❌")
            break
    item.bdd_step_results.append((step.name, status, icon))
