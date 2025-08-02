# conftest.py

import pytest
import os
import html
import platform
from datetime import datetime
from typing import Dict, Any

import pytest_html

# Optimized: Import helpers and utilities from other modules
from utils.csv_reader import load_csv_data
from utils.report_helpers import StepLogCapture, get_report_css, get_report_js
from config.settings import Settings

# --- Pytest Core Hooks ---

def pytest_addoption(parser):
    """Adds custom command-line options for environment and DB configuration."""
    parser.addoption("--env", action="store", default="qa", help="Environment to run tests against")
    parser.addoption("--db-host", action="store", help="Database host")
    parser.addoption("--db-port", action="store", help="Database port")
    parser.addoption("--db-user", action="store", help="Database user")
    parser.addoption("--db-password", action="store", help="Database password")
    parser.addoption("--db-service-name", action="store", help="Oracle service name")

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Configures the test environment and collects metadata for the report."""
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

    from utils import logger_config
    logger_config.setup_logging()
    
    config._metadata = {
        "Python Version": platform.python_version(),
        "Platform": platform.system(),
        "Environment": config.getoption("--env"),
        "Report Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    config._html_report_assets_added = False

def pytest_bdd_generate_tests(metafunc):
    """Dynamically generates parameterized tests from the CSV file."""
    if "test_data_row" in metafunc.fixturenames:
        if "complaint_capture.feature" in metafunc.definition.obj.__scenario__.feature.filename:
            all_test_data = load_csv_data("complaint_data.csv")
            test_case_ids = [row.get("test_case_id", "Unnamed-TC") for row in all_test_data]
            metafunc.parametrize("test_data_row", all_test_data, ids=test_case_ids)

# --- Fixtures ---

@pytest.fixture(scope="session")
def config() -> Settings:
    """
    Loads the application configuration lazily and once per session.
    This fixture ensures that config loading happens *after* pytest_configure
    has set up the necessary environment variables, preventing validation errors.
    """
    from config.loader import load_and_get_config
    return load_and_get_config()

@pytest.fixture(scope="function")
def chatbot_context() -> Dict[str, Any]:
    """Provides a clean dictionary for sharing state within a single test scenario."""
    return {}

# --- HTML Report Generation Hooks ---

def pytest_html_report_title(report):
    """Sets a custom title for the HTML report."""
    report.title = "Chatbot Complaint AI - Interactive BDD Report"

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Modifies the report object to include our custom, expandable step details."""
    outcome = yield
    report = outcome.get_result()
    if report.when == 'call':
        setattr(report, 'report_status', report.outcome)
    if report.when == "call" and hasattr(item, "bdd_step_results"):
        steps_html = ""
        for step_name, status, icon, logs in item.bdd_step_results:
            escaped_logs = html.escape(logs)
            if logs.strip():
                steps_html += f'<details class="step-details"><summary class="step-summary {status.lower()}"><span class="status-icon">{icon}</span> {step_name}</summary><pre class="log-output">{escaped_logs}</pre></details>'
            else:
                steps_html += f'<div class="step-summary {status.lower()}"><span class="status-icon">{icon}</span> {step_name}</div>'
        if steps_html:
            report.extra.append(pytest_html.extras.html(f'<div class="scenario-steps"><h4>Scenario Steps:</h4>{steps_html}</div>'))

def pytest_html_results_table_row(report, cells):
    """Add a class to the results table row for filtering."""
    if hasattr(report, 'report_status'):
        cells[1].attrib['class'] = f'result-{report.report_status}'

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_scenario(request, feature, scenario):
    """Initializes a list to hold step results before each scenario runs."""
    request.node.bdd_step_results = []

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_step(request, step):
    """Before each step, start capturing logs using the helper class."""
    capturer = StepLogCapture()
    request.node.step_log_capturer = capturer
    capturer.__enter__()

@pytest.hookimpl(tryfirst=True)
def pytest_bdd_after_step(request, step, step_result):
    """After each step, stop capturing logs and store the result."""
    item = request.node
    capturer = item.step_log_capturer
    capturer.__exit__(None, None, None)
    logs = capturer.get_logs()
    status, icon = ("Failed", "❌") if step_result.failed else ("Passed", "✅")
    item.bdd_step_results.append((step.name, status, icon, logs))

def pytest_html_results_summary(prefix, summary, postfix):
    """Prepend the metadata and filter controls to the summary section."""
    meta_html = "<h3>Execution Summary</h3><table class='meta-table'>"
    if hasattr(summary.config, '_metadata'):
        for key, value in summary.config._metadata.items():
            meta_html += f"<tr><td>{key}</td><td>{value}</td></tr>"
    meta_html += "</table>"
    
    passed = len(summary.get("passed", []))
    failed = len(summary.get("failed", []))
    skipped = len(summary.get("skipped", []))
    total = passed + failed + skipped

    filter_html = f"""
        <h3>Filter Results</h3>
        <div class="filter-controls">
            <button id="filter-all" class="active" onclick="filterResults('all')">All ({total})</button>
            <button id="filter-passed" onclick="filterResults('passed')">Passed ({passed})</button>
            <button id="filter-failed" onclick="filterResults('failed')">Failed ({failed})</button>
            <button id="filter-skipped" onclick="filterResults('skipped')">Skipped ({skipped})</button>
        </div>
    """
    prefix.extend([pytest_html.extras.html(f"<div class='meta-container'>{meta_html}</div>")])
    prefix.extend([pytest_html.extras.html(f"<div class='filter-container'>{filter_html}</div>")])

def pytest_html_style(report):
    """Injects custom CSS and JavaScript into the report using helper functions."""
    if not getattr(report.config, '_html_report_assets_added', False):
        report.extra.append(pytest_html.extras.html(f"<style>{get_report_css()}</style>"))
        report.extra.append(pytest_html.extras.html(f"<script>{get_report_js()}</script>"))
        report.config._html_report_assets_added = True
