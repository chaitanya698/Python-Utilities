import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .logger_config import get_logger


class BusinessReportGenerator:
    """Generate detailed HTML reports with step-level information."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.step_results = []
        self.test_results = []
    
    def add_step_result(self, step_info: Dict[str, Any]):
        """Add step-level result for reporting."""
        self.step_results.append({
            'timestamp': datetime.now().isoformat(),
            **step_info
        })
    
    def generate_report(
        self, 
        test_results: List[Dict[str, Any]], 
        execution_metadata: Dict[str, Any]
    ) -> str:
        """Generate comprehensive HTML report with step-level details."""
        self.logger.info("Generating detailed business report...")
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_results)
        
        # Build HTML report
        html = self._build_detailed_html_report(metrics, test_results, execution_metadata)
        
        # Save report
        report_path = Path("reports") / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _calculate_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate key metrics from test results."""
        total = len(test_results)
        passed = sum(1 for r in test_results if r.get('status') == 'passed')
        failed = sum(1 for r in test_results if r.get('status') == 'failed')
        skipped = sum(1 for r in test_results if r.get('status') == 'skipped')
        
        durations = [r.get('duration', 0) for r in test_results if r.get('duration')]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'skipped': skipped,
            'pass_rate': (passed / total * 100) if total > 0 else 0,
            'fail_rate': (failed / total * 100) if total > 0 else 0,
            'avg_duration': avg_duration,
            'total_duration': sum(durations)
        }
    
    def _build_detailed_html_report(
        self, 
        metrics: Dict[str, Any], 
        test_results: List[Dict[str, Any]], 
        execution_metadata: Dict[str, Any]
    ) -> str:
        """Build HTML report with expandable step details."""
        
        # Group step results by test
        steps_by_test = {}
        for step in self.step_results:
            test_id = step.get('test_id', 'unknown')
            if test_id not in steps_by_test:
                steps_by_test[test_id] = []
            steps_by_test[test_id].append(step)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Execution Report - {execution_metadata.get('environment', 'Unknown').upper()}</title>
    <style>
        {self._get_enhanced_css()}
    </style>
    <script>
        {self._get_javascript()}
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chatbot Complaint AI - Test Execution Report</h1>
            <div class="subtitle">
                Environment: <span class="env-badge">{execution_metadata.get('environment', 'Unknown').upper()}</span> |
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-value">{metrics['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card success">
                <div class="metric-value">{metrics['passed']}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card danger">
                <div class="metric-value">{metrics['failed']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['pass_rate']:.1f}%</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['avg_duration']:.2f}s</div>
                <div class="metric-label">Avg Duration</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Results with Step Details</h2>
            {self._build_test_details_html(test_results, steps_by_test)}
        </div>
        
        <div class="footer">
            <p>Platform: {execution_metadata.get('platform', 'Unknown')} | 
               Python: {execution_metadata.get('python_version', 'Unknown')} |
               Total Duration: {metrics['total_duration']:.2f}s</p>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _build_test_details_html(self, test_results: List[Dict], steps_by_test: Dict) -> str:
        """Build HTML for test details with expandable steps."""
        html = ""
        
        for test in test_results:
            test_id = test.get('test_id', 'unknown')
            status = test.get('status', 'unknown')
            status_class = f"status-{status}"
            
            html += f"""
            <div class="test-item {status_class}">
                <div class="test-header" onclick="toggleSteps('{test_id}')">
                    <span class="test-name">{test_id}</span>
                    <span class="test-status {status_class}">{status.upper()}</span>
                    <span class="test-duration">{test.get('duration', 0):.2f}s</span>
                    <span class="toggle-icon" id="icon-{test_id}">▶</span>
                </div>
                <div class="test-steps" id="steps-{test_id}" style="display: none;">
            """
            
            steps = steps_by_test.get(test_id, [])
            for step in steps:
                step_status = step.get('status', 'unknown')
                html += f"""
                    <div class="step-item {step_status}">
                        <div class="step-type">{step.get('type', 'step').upper()}</div>
                        <div class="step-description">{step.get('description', 'No description')}</div>
                        <div class="step-status">{step_status}</div>
                        {f'<div class="step-error">{step.get("error")}</div>' if step.get("error") else ''}
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
        
        return html
    
    def _get_enhanced_css(self) -> str:
        """Get enhanced CSS for the report."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 1.1em;
            opacity: 0.95;
        }
        .env-badge {
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: 600;
        }
        .metrics-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 0.9em;
            color: #7f8c8d;
            text-transform: uppercase;
            margin-top: 10px;
        }
        .metric-card.success .metric-value { color: #27ae60; }
        .metric-card.danger .metric-value { color: #e74c3c; }
        .test-results {
            padding: 30px;
        }
        .test-results h2 {
            margin-bottom: 20px;
            color: #2c3e50;
        }
        .test-item {
            margin-bottom: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .test-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f8f9fa;
            cursor: pointer;
            transition: background 0.2s;
        }
        .test-header:hover {
            background: #e9ecef;
        }
        .test-name {
            font-weight: 600;
            flex-grow: 1;
        }
        .test-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 0 10px;
        }
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        .status-skipped {
            background: #fff3cd;
            color: #856404;
        }
        .test-duration {
            color: #6c757d;
            margin-right: 10px;
        }
        .toggle-icon {
            transition: transform 0.3s;
        }
        .toggle-icon.expanded {
            transform: rotate(90deg);
        }
        .test-steps {
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        .step-item {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            border-bottom: 1px solid #f0f0f0;
        }
        .step-item:last-child {
            border-bottom: none;
        }
        .step-type {
            background: #007bff;
            color: white;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 0.75em;
            margin-right: 15px;
            min-width: 60px;
            text-align: center;
        }
        .step-description {
            flex-grow: 1;
            color: #495057;
        }
        .step-status {
            font-size: 0.85em;
            padding: 3px 10px;
            border-radius: 4px;
        }
        .step-item.passed .step-status {
            background: #d4edda;
            color: #155724;
        }
        .step-item.failed .step-status {
            background: #f8d7da;
            color: #721c24;
        }
        .step-error {
            width: 100%;
            margin-top: 10px;
            padding: 10px;
            background: #f8d7da;
            color: #721c24;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }
        """
    
    def _get_javascript(self) -> str:
        """Get JavaScript for interactive features."""
        return """
        function toggleSteps(testId) {
            const stepsDiv = document.getElementById('steps-' + testId);
            const icon = document.getElementById('icon-' + testId);
            
            if (stepsDiv.style.display === 'none') {
                stepsDiv.style.display = 'block';
                icon.classList.add('expanded');
                icon.textContent = '▼';
            } else {
                stepsDiv.style.display = 'none';
                icon.classList.remove('expanded');
                icon.textContent = '▶';
            }
        }
        """