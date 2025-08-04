import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from io import BytesIO

from .logger_config import get_logger


class BusinessReportGenerator:
    """Generate executive-friendly HTML reports with visualizations."""
    
    def __init__(self, template_dir: Path = Path("reports")):
        self.template_dir = template_dir
        self.logger = get_logger(__name__)
    
    def generate_report(
        self, 
        test_results: List[Dict[str, Any]], 
        execution_metadata: Dict[str, Any]
    ) -> str:
        """Generate comprehensive HTML report."""
        self.logger.info("Generating business report...")
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_results)
        
        # Generate visualizations
        charts = self._generate_charts(metrics)
        
        # Build HTML report
        html = self._build_html_report(metrics, charts, test_results, execution_metadata)
        
        # Save report
        report_path = Path("reports") / f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        self.logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _calculate_metrics(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate key business metrics."""
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
    
    def _generate_charts(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate charts as base64 encoded images."""
        charts = {}
        
        # Pie chart for test status distribution
        if metrics['total_tests'] > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            labels = []
            sizes = []
            colors = []
            
            if metrics['passed'] > 0:
                labels.append(f"Passed ({metrics['passed']})")
                sizes.append(metrics['passed'])
                colors.append('#28a745')
            
            if metrics['failed'] > 0:
                labels.append(f"Failed ({metrics['failed']})")
                sizes.append(metrics['failed'])
                colors.append('#dc3545')
            
            if metrics['skipped'] > 0:
                labels.append(f"Skipped ({metrics['skipped']})")
                sizes.append(metrics['skipped'])
                colors.append('#ffc107')
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title('Test Execution Status Distribution', fontsize=16, fontweight='bold')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            charts['status_pie'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        return charts
    
    def _build_html_report(
        self, 
        metrics: Dict[str, Any], 
        charts: Dict[str, str],
        test_results: List[Dict[str, Any]], 
        execution_metadata: Dict[str, Any]
    ) -> str:
        """Build the complete HTML report."""
        
        # Read CSS file if exists
        css_content = ""
        css_file = Path("reports/style.css")
        if css_file.exists():
            with open(css_file, 'r') as f:
                css_content = f.read()
        else:
            css_content = self._get_default_css()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot Complaint AI - Test Execution Report</title>
    <style>
        {css_content}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Chatbot Complaint AI - Test Execution Report</h1>
            <div class="subtitle">Automated Test Results and Performance Metrics</div>
        </div>
        
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            <p style="margin-top: 15px; font-size: 1.1em; color: #555;">
                This report provides a comprehensive overview of the automated test execution for the 
                Chatbot Complaint AI system. The tests validate the end-to-end complaint capture workflow,
                ensuring system reliability and compliance with business requirements.
            </p>
            
            <div class="metrics-grid" style="margin-top: 30px;">
                <div class="metric-card">
                    <div class="metric-label">Total Tests</div>
                    <div class="metric-value">{metrics['total_tests']}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Pass Rate</div>
                    <div class="metric-value {'success' if metrics['pass_rate'] >= 90 else 'warning' if metrics['pass_rate'] >= 70 else 'danger'}">
                        {metrics['pass_rate']:.1f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Failed Tests</div>
                    <div class="metric-value {'danger' if metrics['failed'] > 0 else 'success'}">
                        {metrics['failed']}
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">Avg Duration</div>
                    <div class="metric-value">{metrics['avg_duration']:.2f}s</div>
                </div>
            </div>
        </div>
        """
        
        if charts.get('status_pie'):
            html += f"""
        <div class="chart-container">
            <h3>Test Status Distribution</h3>
            <img src="data:image/png;base64,{charts['status_pie']}" alt="Status Distribution Chart">
        </div>
        """
        
        html += """
        <div class="details-section">
            <h3>Detailed Test Results</h3>
            <table>
                <thead>
                    <tr>
                        <th>Test Case ID</th>
                        <th>Description</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add test results
        for result in test_results:
            status_class = f"status-{result.get('status', 'unknown')}"
            html += f"""
                    <tr>
                        <td><strong>{result.get('test_id', 'N/A')}</strong></td>
                        <td>{result.get('description', 'N/A')}</td>
                        <td><span class="status-badge {status_class}">{result.get('status', 'unknown').upper()}</span></td>
                        <td>{result.get('duration', 0):.2f}s</td>
                        <td>{result.get('timestamp', 'N/A')}</td>
                    </tr>
            """
        
        html += f"""
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Environment: {execution_metadata.get('environment', 'Unknown')} | 
            Platform: {execution_metadata.get('platform', 'Unknown')} | 
            Framework Version: 1.0.0</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def _get_default_css(self) -> str:
        """Get default CSS if style.css is not available."""
        return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa;
            color: #2c3e50;
            line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; border-radius: 10px;
            margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; }
        .header .subtitle { font-size: 1.2em; opacity: 0.9; }
        .executive-summary {
            background: white; padding: 30px; border-radius: 10px;
            margin-bottom: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px; margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 25px; border-radius: 10px; text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        }
        .metric-value {
            font-size: 2.5em; font-weight: bold; color: #2c3e50; margin: 10px 0;
        }
        .metric-label {
            font-size: 1em; color: #7f8c8d; text-transform: uppercase;
            letter-spacing: 1px;
        }
        .success { color: #27ae60; }
        .danger { color: #e74c3c; }
        .warning { color: #f39c12; }
        .chart-container {
            background: white; padding: 30px; border-radius: 10px;
            margin-bottom: 30px; box-shadow: 0 5px 20px rgba(0,0,0,0.05);
            text-align: center;
        }
        .details-section {
            background: white; padding: 30px; border-radius: 10px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        }
        table {
            width: 100%; border-collapse: collapse; margin-top: 20px;
        }
        th {
            background-color: #f8f9fa; padding: 15px; text-align: left;
            font-weight: 600; color: #495057; border-bottom: 2px solid #dee2e6;
        }
        td { padding: 12px 15px; border-bottom: 1px solid #dee2e6; }
        tr:hover { background-color: #f8f9fa; }
        .status-badge {
            display: inline-block; padding: 5px 12px; border-radius: 20px;
            font-size: 0.85em; font-weight: 600;
        }
        .status-passed { background-color: #d4edda; color: #155724; }
        .status-failed { background-color: #f8d7da; color: #721c24; }
        .status-skipped { background-color: #fff3cd; color: #856404; }
        .footer {
            text-align: center; padding: 30px; color: #6c757d; font-size: 0.9em;
        }
        """