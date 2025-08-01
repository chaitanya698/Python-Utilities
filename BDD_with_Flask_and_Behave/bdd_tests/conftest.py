# bdd_tests/conftest.py

import pytest
import pytest_html
import logging
from typing import Dict, Any

from bdd_tests.utils.api_service import ChatbotAPIService
from bdd_tests.utils.db_utils import DBUtils

logger = logging.getLogger(__name__)

# --- Service Fixtures ---

@pytest.fixture(scope="session")
def api_service() -> ChatbotAPIService:
    """Provides a session-scoped instance of the ChatbotAPIService."""
    return ChatbotAPIService()

@pytest.fixture(scope="function")
def db_utils() -> DBUtils:
    """Provides a function-scoped DB utility that handles connection setup and teardown."""
    db = DBUtils()
    try:
        db.connect()
        yield db
    finally:
        db.disconnect()

# --- Context Fixture ---

@pytest.fixture(scope="function")
def chatbot_context() -> Dict[str, Any]:
    """
    Provides a clean dictionary for sharing state within a single scenario.
    This is reset for each scenario to ensure test isolation.
    """
    return {}

# --- Pytest HTML Reporting Hooks ---

def pytest_html_report_title(report):
    """Sets a custom title for the HTML report."""
    report.title = "Chatbot Complaint AI - BDD Test Report"

def pytest_configure(config):
    """Adds project metadata to the HTML report header."""
    config.stash['metadata'] = {
        'Project': 'Space Complaint AI',
        'Framework': 'Pytest-BDD',
    }

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Adds detailed BDD steps with status icons to the HTML report."""
    outcome = yield
    report = outcome.get_result()
    
    # We only want to add details to the 'call' phase of the test
    if report.when == "call":
        # Add step details if they were captured by our custom hook
        if hasattr(item, "bdd_step_results"):
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
def pytest_bdd_after_step(request, step, step_func, step_func_args):
    """
    Records the result of each step right after it runs.
    This is a more reliable way to capture step status for reporting.
    """
    item = request.node
    # Find the report for the 'call' phase of the current test item
    try:
        call_report = next(
            rep for rep in reversed(item.session.reports) 
            if rep.nodeid == item.nodeid and rep.when == "call"
        )
        status, icon = ("Passed", "✅") if call_report.passed else ("Failed", "❌")
    except (StopIteration, AttributeError):
        # This can happen if the step fails during setup
        status, icon = ("Failed", "❌")

    # If the step itself failed, override status
    if call_report.outcome != "passed":
        status, icon = ("Failed", "❌")

    item.bdd_step_results.append((step.name, status, icon))
