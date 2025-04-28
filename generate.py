"""
generate.py - Enhanced report builder
------------------
Improvements:
* Better visualization of nested tables
* Enhanced side-by-side diff for tables
* Improved summary statistics
* Interactive UI elements
* Support for multi-page tables
"""
import os
import re
import logging
import json
from datetime import datetime
from typing import Dict, Optional, List, Any
from jinja2 import Template

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    # ─── public ───────────────────────────────────────────────────
    def generate_html_report(self, results: Dict,
                             pdf1_name: str, pdf2_name: str,
                             metadata: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive HTML comparison report.
        
        Args:
            results: The comparison results
            pdf1_name: Name of the first PDF
            pdf2_name: Name of the second PDF
            metadata: Additional metadata to include
            
        Returns:
            Path to the generated HTML report
        """
        logger.info("Generating HTML comparison report")
        
        # Calculate summary statistics
        summary = self._generate_summary(results)
        
        # Render HTML template
        html = self._render(results, pdf1_name, pdf2_name, metadata or {}, summary)
        
        # Create unique filename
        filename = f"compare_{self._safe(pdf1_name)}_vs_{self._safe(pdf2_name)}_{datetime.now():%Y%m%d_%H%M%S}.html"
        output_path = os.path.join(self.output_dir, filename)
        
        # Write the report to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        logger.info(f"HTML report written to {output_path}")
        return output_path

    # ─── helpers ──────────────────────────────────────────────────
    def _render(self, results, pdf1_name, pdf2_name, metadata, summary):
        """Render the HTML template with the provided data."""
        template = Template(self._template_str(), autoescape=True)
        return template.render(
            results=results,
            pdf1_name=pdf1_name,
            pdf2_name=pdf2_name,
            metadata=metadata,
            summary=summary,
            title=f"PDF Comparison: {pdf1_name} vs {pdf2_name}",
            comparison_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate comprehensive summary statistics."""
        pages_with_diff = 0
        total_text = 0
        total_table = 0
        matched_tables = 0
        moved_tables = 0
        modified_tables = 0
        deleted_tables = 0
        inserted_tables = 0
        
        # Analyze differences by page
        for page_num, page in results.get("pages", {}).items():
            page_has_diff = False
            
            # Text differences
            text_diffs = [d for d in page.get("text_differences", []) if d.get("status") != "equal"]
            if text_diffs:
                page_has_diff = True
                total_text += len(text_diffs)
            
            # Table differences
            for table in page.get("table_differences", []):
                status = table.get("status", "")
                
                if status == "matched":
                    matched_tables += 1
                elif status == "moved":
                    moved_tables += 1
                    # Moving doesn't count as a difference
                elif status == "modified":
                    modified_tables += 1
                    total_table += 1
                    page_has_diff = True
                elif status == "deleted":
                    deleted_tables += 1
                    total_table += 1
                    page_has_diff = True
                elif status == "inserted":
                    inserted_tables += 1
                    total_table += 1
                    page_has_diff = True
            
            if page_has_diff:
                pages_with_diff += 1
        
        # Calculate total tables processed
        total_tables = matched_tables + moved_tables + modified_tables + deleted_tables + inserted_tables
        
        # Calculate percentages
        total_pages = results.get("max_pages", 0)
        page_diff_percent = round(100 * pages_with_diff / max(1, total_pages), 2)
        table_diff_percent = round(100 * (modified_tables + deleted_tables + inserted_tables) / max(1, total_tables), 2) if total_tables > 0 else 0
        
        return {
            "total_pages": total_pages,
            "pages_with_differences": pages_with_diff,
            "percentage_changed_pages": page_diff_percent,
            
            "total_text_differences": total_text,
            "total_table_differences": total_table,
            "total_differences": total_text + total_table,
            
            "table_statistics": {
                "matched": matched_tables,
                "moved": moved_tables,
                "modified": modified_tables,
                "deleted": deleted_tables,
                "inserted": inserted_tables,
                "total": total_tables,
                "percentage_changed": table_diff_percent
            }
        }

    @staticmethod
    def _safe(name):
        """Create a safe filename from the document name."""
        return re.sub(r"[^\w\-\.]", "_", name)[:50]

    # ─── HTML template ────────────────────────────────────────────
    @staticmethod
    def _template_str() -> str:
        """Return the HTML template for the comparison report."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        /* Base styles */
        :root {
            --primary-color: #4285F4;
            --success-color: #34A853;
            --warning-color: #FBBC05;
            --danger-color: #EA4335;
            --light-color: #F8F9FA;
            --dark-color: #212529;
            --deleted-bg: #FFEDED;
            --inserted-bg: #EDFFF0;
            --modified-bg: #FCF8E3;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--dark-color);
            margin: 0;
            padding: 0;
            background-color: #F5F5F5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            min-height: 100vh;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
        }
        
        h1, h2, h3 {
            margin-top: 0;
        }
        
        footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            text-align: center;
            color: #777;
        }
        
        /* Summary card */
        .summary-card {
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .summary-title {
            font-size: 1.4rem;
            margin-bottom: 15px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .stat-card {
            background-color: var(--light-color);
            border-radius: 5px;
            padding: 15px;
            text-align: center;
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }
        
        .stat-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .stat-card.success .stat-value { color: var(--success-color); }
        .stat-card.warning .stat-value { color: var(--warning-color); }
        .stat-card.danger .stat-value { color: var(--danger-color); }
        
        /* Table styles */
        .table-stats {
            margin-top: 20px;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }
        
        .table-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        
        .badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            display: inline-flex;
            align-items: center;
        }
        
        .badge::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .badge.matched {
            background-color: rgba(52, 168, 83, 0.15);
            color: #2e7d32;
        }
        .badge.matched::before { background-color: #2e7d32; }
        
        .badge.moved {
            background-color: rgba(66, 133, 244, 0.15);
            color: #1976d2;
        }
        .badge.moved::before { background-color: #1976d2; }
        
        .badge.modified {
            background-color: rgba(251, 188, 5, 0.15);
            color: #f57f17;
        }
        .badge.modified::before { background-color: #f57f17; }
        
        .badge.deleted {
            background-color: rgba(234, 67, 53, 0.15);
            color: #c62828;
        }
        .badge.deleted::before { background-color: #c62828; }
        
        .badge.inserted {
            background-color: rgba(156, 39, 176, 0.15);
            color: #7b1fa2;
        }
        .badge.inserted::before { background-color: #7b1fa2; }
        
        /* Page sections */
        .page-section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .page-section h2 {
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .page-controls {
            display: flex;
            gap: 10px;
        }
        
        .page-badge {
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: normal;
        }
        
        .page-badge.has-diff {
            background-color: var(--warning-color);
            color: white;
        }
        
        .page-badge.no-diff {
            background-color: var(--success-color);
            color: white;
        }
        
        /* Diff headers */
        .diff-header {
            background-color: #f5f5f5;
            padding: 10px 15px;
            margin: 15px 0 10px;
            border-radius: 4px;
            font-weight: 500;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .diff-header.deleted {
            background-color: var(--deleted-bg);
            color: var(--danger-color);
            border-left: 4px solid var(--danger-color);
        }
        
        .diff-header.inserted {
            background-color: var(--inserted-bg);
            color: var(--success-color);
            border-left: 4px solid var(--success-color);
        }
        
        .diff-header.modified {
            background-color: var(--modified-bg);
            color: var(--warning-color);
            border-left: 4px solid var(--warning-color);
        }
        
        /* Text differences */
        .text-diff-line {
            padding: 5px 10px;
            margin: 2px 0;
            border-radius: 3px;
            font-family: monospace;
            white-space: pre-wrap;
            display: flex;
        }
        
        .line-number {
            color: #999;
            margin-right: 15px;
            user-select: none;
            min-width: 30px;
            text-align: right;
        }
        
        .text-content {
            flex: 1;
        }
        
        .text-diff-line.deleted {
            background-color: var(--deleted-bg);
        }
        
        .text-diff-line.inserted {
            background-color: var(--inserted-bg);
        }
        
        .text-diff-line.changed,
        .text-diff-line.modified {
            background-color: var(--modified-bg);
        }
        
        /* Inline text diff */
        .inline-diff {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 5px 0;
            padding: 10px;
            border-radius: 4px;
            background-color: #f8f8f8;
        }
        
        .diff-old, .diff-new {
            padding: 8px;
            border-radius: 3px;
        }
        
        .diff-old {
            background-color: rgba(234, 67, 53, 0.08);
        }
        
        .diff-new {
            background-color: rgba(52, 168, 83, 0.08);
        }
        
        .diff-deleted {
            background-color: var(--deleted-bg);
            text-decoration: line-through;
            color: var(--danger-color);
            padding: 0 2px;
            border-radius: 2px;
        }
        
        .diff-inserted {
            background-color: var(--inserted-bg);
            color: var(--success-color);
            padding: 0 2px;
            border-radius: 2px;
        }
        
        /* Table differences */
        .table-diff-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
            overflow-x: auto;
        }
        
        .table-diff-left, .table-diff-right {
            max-width: 100%;
            overflow-x: auto;
        }
        
        .diff-table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #ddd;
            font-size: 0.9rem;
        }
        
        .diff-table td, .diff-table th {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        .diff-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        .diff-table tr:hover {
            background-color: #f1f1f1;
        }
        
        .diff-table.deleted {
            background-color: var(--deleted-bg);
            border: 1px solid var(--danger-color);
        }
        
        .diff-table.inserted {
            background-color: var(--inserted-bg);
            border: 1px solid var(--success-color);
        }
        
        .diff-table td.deleted {
            background-color: var(--deleted-bg);
            color: var(--danger-color);
        }
        
        .diff-table td.inserted {
            background-color: var(--inserted-bg);
            color: var(--success-color);
        }
        
        /* Nested tables */
        .nested-table-indicator {
            padding: 2px 6px;
            background-color: rgba(0,0,0,0.1);
            border-radius: 3px;
            font-size: 0.8rem;
            margin-left: 10px;
        }
        
        .nested-table-container {
            margin-left: 20px;
            border-left: 2px solid var(--primary-color);
            padding-left: 20px;
        }
        
        /* Badges */
        .moved-badge {
            background: var(--primary-color);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.8rem;
            display: inline-block;
        }
        
        /* Collapsible sections */
        .collapsible {
            cursor: pointer;
        }
        
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
        }
        
        .expanded .collapsible-content {
            max-height: 2000px; /* Large enough to show content */
        }
        
        .toggle-icon {
            display: inline-block;
            transition: transform 0.3s;
            margin-right: 5px;
        }
        
        .expanded .toggle-icon {
            transform: rotate(90deg);
        }
        
        /* Metadata */
        .metadata {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            background-color: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            font-size: 0.9rem;
        }
        
        .metadata-item {
            display: flex;
        }
        
        .metadata-label {
            font-weight: 500;
            margin-right: 10px;
            color: #666;
        }
        
        .metadata-value {
            flex: 1;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .summary-grid {
                grid-template-columns: 1fr 1fr;
            }
            
            .table-diff-container {
                grid-template-columns: 1fr;
            }
            
            .metadata {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 480px) {
            .summary-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <div>Comparison Date: {{ comparison_date }}</div>
        </header>
        
        <!-- Summary Section -->
        <section class="summary-card">
            <h2 class="summary-title">Summary</h2>
            
            <div class="summary-grid">
                <div class="stat-card {{ 'danger' if summary.pages_with_differences > 0 else 'success' }}">
                    <div class="stat-label">Pages with Differences</div>
                    <div class="stat-value">{{ summary.pages_with_differences }} / {{ summary.total_pages }}</div>
                </div>
                
                <div class="stat-card {{ 'danger' if summary.percentage_changed_pages > 20 else 'warning' if summary.percentage_changed_pages > 5 else 'success' }}">
                    <div class="stat-label">% of Pages Changed</div>
                    <div class="stat-value">{{ summary.percentage_changed_pages }}%</div>
                </div>
                
                <div class="stat-card {{ 'danger' if summary.total_text_differences > 20 else 'warning' if summary.total_text_differences > 5 else 'success' }}">
                    <div class="stat-label">Text Differences</div>
                    <div class="stat-value">{{ summary.total_text_differences }}</div>
                </div>
                
                <div class="stat-card {{ 'danger' if summary.total_table_differences > 10 else 'warning' if summary.total_table_differences > 2 else 'success' }}">
                    <div class="stat-label">Table Differences</div>
                    <div class="stat-value">{{ summary.total_table_differences }}</div>
                </div>
            </div>
            
            <div class="table-stats">
                <h3>Table Changes</h3>
                <div class="table-badges">
                    <div class="badge matched">Matched: {{ summary.table_statistics.matched }}</div>
                    <div class="badge moved">Moved: {{ summary.table_statistics.moved }}</div>
                    <div class="badge modified">Modified: {{ summary.table_statistics.modified }}</div>
                    <div class="badge deleted">Deleted: {{ summary.table_statistics.deleted }}</div>
                    <div class="badge inserted">Inserted: {{ summary.table_statistics.inserted }}</div>
                </div>
            </div>
            
            <!-- Metadata -->
            {% if metadata %}
            <div class="metadata">
                {% for key, value in metadata.items() %}
                <div class="metadata-item">
                    <div class="metadata-label">{{ key }}:</div>
                    <div class="metadata-value">{{ value }}</div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </section>
        
        <!-- Page by Page Comparison -->
        {% for page_num in range(1, results.max_pages + 1) %}
            {% set page = results.pages.get(page_num, {}) %}
            {% set has_differences = false %}
            
            {% if page.text_differences %}
                {% for diff in page.text_differences %}
                    {% if diff.status != 'equal' %}
                        {% set has_differences = true %}
                    {% endif %}
                {% endfor %}
            {% endif %}
            
            {% if page.table_differences %}
                {% for diff in page.table_differences %}
                    {% if diff.status not in ['matched', 'moved'] %}
                        {% set has_differences = true %}
                    {% endif %}
                {% endfor %}
            {% endif %}
            
            <section id="page-{{ page_num }}" class="page-section">
                <h2>
                    Page {{ page_num }}
                    <div class="page-controls">
                        {% if has_differences %}
                            <span class="page-badge has-diff">Differences Found</span>
                        {% else %}
                            <span class="page-badge no-diff">No Differences</span>
                        {% endif %}
                    </div>
                </h2>
                
                <!-- Text Differences -->
                {% if page.text_differences %}
                    <div class="text-differences">
                        <div class="diff-header">Text Content</div>
                        {% for diff in page.text_differences %}
                            {% if diff.status == 'equal' %}
                                <div class="text-diff-line">
                                    <span class="line-number">{{ diff.line_num1 + 1 }}</span>
                                    <span class="text-content">{{ diff.text1 }}</span>
                                </div>
                            {% elif diff.status == 'deleted' %}
                                <div class="text-diff-line deleted">
                                    <span class="line-number">{{ diff.line_num1 + 1 }}</span>
                                    <span class="text-content">{{ diff.text1 }}</span>
                                </div>
                            {% elif diff.status == 'inserted' %}
                                <div class="text-diff-line inserted">
                                    <span class="line-number">{{ diff.line_num2 + 1 }}</span>
                                    <span class="text-content">{{ diff.text2 }}</span>
                                </div>
                            {% elif diff.status in ['changed', 'modified'] %}
                                {% if diff.diff_html %}
                                    {{ diff.diff_html|safe }}
                                {% else %}
                                    <div class="text-diff-line changed">
                                        <span class="line-number">{{ diff.line_num1 + 1 if diff.line_num1 is not none else '-' }}</span>
                                        <span class="text-content">{{ diff.text1 }}</span>
                                    </div>
                                    <div class="text-diff-line changed">
                                        <span class="line-number">{{ diff.line_num2 + 1 if diff.line_num2 is not none else '-' }}</span>
                                        <span class="text-content">{{ diff.text2 }}</span>
                                    </div>
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                    </div>
                {% endif %}
                
                <!-- Table Differences -->
                {% for t in page.table_differences %}
                    {% if t.status == 'matched' %}
                        <div class="diff-header">
                            Table {{ t.table_id1 or t.table_id2 }} (Identical)
                            {% if t.has_nested_tables %}
                                <span class="nested-table-indicator">Contains nested tables</span>
                            {% endif %}
                        </div>
                    {% elif t.status == 'moved' %}
                        <div class="diff-header">
                            Table {{ t.table_id1 or t.table_id2 }}
                            <span class="moved-badge">Moved from page {{ t.page1 }} to {{ t.page2 }}</span>
                            {% if t.has_nested_tables %}
                                <span class="nested-table-indicator">Contains nested tables</span>
                            {% endif %}
                        </div>
                    {% elif t.status == 'modified' %}
                        <div class="diff-header modified">
                            Table {{ t.table_id1 or t.table_id2 }} (Modified - {{ t.differences }} differences)
                            {% if t.has_nested_tables %}
                                <span class="nested-table-indicator">Contains nested tables</span>
                            {% endif %}
                        </div>
                        {{ t.diff_html|safe }}
                    {% elif t.status == 'deleted' %}
                        <div class="diff-header deleted">
                            Table only in first document
                            {% if t.has_nested_tables %}
                                <span class="nested-table-indicator">Contains nested tables</span>
                            {% endif %}
                        </div>
                        {{ t.diff_html|safe }}
                    {% elif t.status == 'inserted' %}
                        <div class="diff-header inserted">
                            Table only in second document
                            {% if t.has_nested_tables %}
                                <span class="nested-table-indicator">Contains nested tables</span>
                            {% endif %}
                        </div>
                        {{ t.diff_html|safe }}
                    {% endif %}
                {% endfor %}
            </section>
        {% endfor %}
        
        <footer>
            Report generated on {{ comparison_date }} | PDF Comparison Tool v2.0
        </footer>
    </div>
    
    <script>
        // Initialize collapsible sections
        document.addEventListener('DOMContentLoaded', function() {
            // Make diff headers collapsible
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(item) {
                item.addEventListener('click', function() {
                    this.classList.toggle('expanded');
                });
            });
            
            // Add jump-to-page functionality
            const pageLinks = document.querySelectorAll('.page-link');
            pageLinks.forEach(function(link) {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href');
                    document.querySelector(targetId).scrollIntoView({
                        behavior: 'smooth'
                    });
                });
            });
        });
    </script>
</body>
</html>
"""