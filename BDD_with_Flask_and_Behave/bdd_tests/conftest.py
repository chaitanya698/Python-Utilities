import pytest
import logging
import pytest_html
from bdd_tests.utils.api_service import ChatbotAPIService
from config.loader import config

# Ensure logging is set up before tests run
from utils.logger_config import get_logger
get_logger(__name__)


# --- Fixtures to provide services to tests ---

@pytest.fixture(scope="session")
def api_service():
    """Provides a session-scoped instance of the ChatbotAPIService."""
    return ChatbotAPIService(config)

@pytest.fixture
def chatbot_context():
    """Provides a clean dictionary for sharing state within a single scenario."""
    return {}


# --- Hooks for Richer HTML Reporting ---

def pytest_html_report_title(report):
    """Set a custom title for the HTML report."""
    report.title = "Chatbot Complaint Flow - Test Report"

def pytest_configure(config):
    """Add metadata to the report header."""
    config._metadata['Project'] = 'Space Complaint AI'
    config._metadata['Framework'] = 'Pytest-BDD'
    config._metadata['Python'] = '3.12.5' # Example version

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """
    Hook to access test outcomes and add extra details to the HTML report.
    This is where we add the detailed step information.
    """
    outcome = yield
    report = outcome.get_result()
    
    # We only want to modify the report for the 'call' phase
    if report.when == "call":
        # Attempt to get the detailed steps from the test item (scenario)
        try:
            scenario = item.function.__scenario__
            steps_details = []
            if hasattr(item, 'step_results'):
                for step, status, icon in item.step_results:
                    steps_details.append(f"<div class='step-detail'><span class='step-status'>{icon}</span> <span class='step-keyword'>{step.keyword}</span> {step.name}</div>")
            
            # Add the formatted steps to the report
            report.extra = getattr(report, 'extra', [])
            report.extra.append(pytest_html.extras.html(f"<h4>Scenario Steps:</h4>{''.join(steps_details)}"))
        except AttributeError:
            # This test is not a BDD scenario, so we do nothing
            pass

@pytest.bdd_before_scenario
def setup_scenario_tracking(request):
    """Initialize a list to hold step results for the current scenario."""
    item = request.node
    item.step_results = []

@pytest.bdd_after_step
def record_step_result(request, step, step_func):
    """Record the result of each step to be used in the final report."""
    item = request.node
    # Find the report for the current test item
    try:
        # This logic finds the most recent report for the 'call' phase of the current node
        report = next(rep for rep in reversed(item.session.reports) if rep.nodeid == item.nodeid and rep.when == 'call')
        if report.passed:
            status, icon = "PASSED", "✅"
        else:
            status, icon = "FAILED", "❌"
    except (StopIteration, AttributeError):
        status, icon = "SKIPPED", "➖"
        
    item.step_results.append((step, status, icon))