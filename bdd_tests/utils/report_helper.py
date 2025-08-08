# utils/report_helpers.py

import logging
import io

class StepLogCapture:
    """A context manager to capture log output for a single BDD step."""
    def __init__(self):
        self.log_stream = io.StringIO()
        # A specific formatter for the report to keep it clean and concise
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%H:%M:%S"
        )
        self.handler = logging.StreamHandler(self.log_stream)
        self.handler.setFormatter(formatter)

    def __enter__(self):
        """Adds the handler to the root logger to start capturing."""
        logging.getLogger().addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Removes the handler to stop capturing."""
        logging.getLogger().removeHandler(self.handler)

    def get_logs(self) -> str:
        """Returns the captured logs as a string."""
        return self.log_stream.getvalue()

def get_report_css() -> str:
    """Returns the CSS string for styling the interactive HTML report."""
    return """
        .report-container { max-width: 1200px; margin: auto; padding: 20px; background-color: #fff; border: 1px solid #e0e0e0; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        .meta-container, .filter-container { margin-bottom: 25px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }
        .meta-table { width: 100%; border-collapse: collapse; }
        .meta-table td { padding: 8px; border-bottom: 1px solid #f0f0f0; }
        .meta-table td:first-child { font-weight: bold; color: #555; width: 150px; }
        .filter-controls button { background-color: #007bff; color: white; border: none; padding: 8px 15px; margin-right: 10px; border-radius: 5px; cursor: pointer; font-size: 14px; transition: background-color: 0.2s; }
        .filter-controls button:hover { background-color: #0056b3; }
        .filter-controls button.active { background-color: #28a745; }
        .scenario-steps { font-family: monospace; margin-top: 15px; }
        .step-details { border-left: 3px solid #ccc; margin-bottom: 5px; padding-left: 10px; }
        .step-summary { cursor: pointer; padding: 5px; display: block; border-radius: 3px; }
        .step-summary:hover { background-color: #f0f0f0; }
        .step-summary.failed { color: #c00; font-weight: bold; }
        details > .step-summary.failed { border-left-color: #c00; }
        .step-summary.passed { color: #080; }
        details > .step-summary.passed { border-left-color: #080; }
        .status-icon { margin-right: 10px; }
        .log-output { background-color: #f5f5f5; border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin-top: 5px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.9em; color: #333; }
        td.result-passed { background-color: #e6ffed !important; }
        td.result-failed { background-color: #ffe6e6 !important; }
        td.result-skipped { background-color: #fff3cd !important; }
    """

def get_report_js() -> str:
    """Returns the JavaScript string for the report's filter functionality."""
    return """
    function filterResults(status) {
        const rows = document.querySelectorAll('#results-table tr');
        document.querySelectorAll('.filter-controls button').forEach(btn => btn.classList.remove('active'));
        document.getElementById(`filter-${status}`).classList.add('active');

        rows.forEach((row, index) => {
            if (index === 0) return; // Keep header row visible
            const statusCell = row.querySelector('td:nth-child(2)');
            if (!statusCell) return;

            if (status === 'all' || statusCell.classList.contains(`result-${status}`)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }
    """
