# bdd_tests/conftest.py

import pytest
import pytest_html
import logging
import os
from typing import Dict, Any

from bdd_tests.utils.api_service import ChatbotAPIService
from bdd_tests.utils.db_utils import DBUtils

logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    """Adds the --env command-line option to pytest."""
    parser.addoption(
        "--env",
        action="store",
        default="qa",
        help="Specify the test environment to run against (e.g., dev, qa)"
    )

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """
    This hook runs after command-line options are parsed.
    It sets the TEST_ENV os variable, which the settings module will then use.
    """
    env = config.getoption("--env")
    os.environ["TEST_ENV"] = env

# --- Service Fixtures ---

@pytest.fixture(scope="session")
def api_service() -> ChatbotAPIService:
    """Provides a session-scoped instance of the ChatbotAPIService."""
    return ChatbotAPIService()

@pytest.fixture(scope="session")
def db_utils() -> DBUtils:
    """Provides a session-scoped DB utility instance."""
    return DBUtils()

# --- Context Fixture ---

@pytest.fixture(scope="function")
def chatbot_context() -> Dict[str, Any]:
    """Provides a clean dictionary for sharing state within a single scenario."""
    return {}

# --- Pytest HTML Reporting Hooks ---

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
