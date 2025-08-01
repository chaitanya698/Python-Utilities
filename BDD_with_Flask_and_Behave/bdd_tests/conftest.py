# bdd_tests/conftest.py

import pytest
import pytest_html
import logging
from typing import Dict, Any

from bdd_tests.utils.api_service import ChatbotAPIService
from bdd_tests.utils.db_utils import DBUtils

logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def api_service() -> ChatbotAPIService:
    return ChatbotAPIService()

@pytest.fixture(scope="session")
def db_utils() -> DBUtils:
    return DBUtils()

@pytest.fixture(scope="function")
def chatbot_context() -> Dict[str, Any]:
    return {}

def pytest_html_report_title(report):
    report.title = "Chatbot Complaint AI - BDD Test Report"

def pytest_configure(config):
    config.stash['metadata'] = {
        'Project': 'Space Complaint AI',
        'Framework': 'Pytest-BDD',
    }

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        if hasattr(item, "bdd_step_results"):
            steps_html = "".join(
                f'<div><span class="status {status.lower()}">{icon}</span> {step_name}</div>'
                for step_name, status, icon in item.bdd_step_results
            )
            report.extra.append(pytest_html.extras.html(f"<h4>Scenario Steps:</h4>{steps_html}"))

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_scenario(request, feature, scenario):
    request.node.bdd_step_results = []

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_after_step(request, step, step_func, step_func_args):
    item = request.node
    status, icon = ("Passed", "✅")
    for rep in item.session.reports:
        if rep.when == "call" and rep.nodeid == item.nodeid and not rep.passed:
            status, icon = ("Failed", "❌")
            break
    item.bdd_step_results.append((step.name, status, icon))
