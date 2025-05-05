
import os
import re
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple, Set
from collections import defaultdict
from jinja2 import Template

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.table_registry = {}

    # ─── public ───────────────────────────────────────────────────
    def generate_html_report(self, results: Dict,
                             pdf1_name: str, pdf2_name: str,
                             metadata: Optional[Dict] = None) -> str:
        """
        Generate a comprehensive HTML comparison report with fixed visual distinction.
        """
        logger.info("Generating HTML comparison report")
        
        # Build table registry for nested table visualization
        self._build_table_registry(results)
        
        # Pre-process results to enhance nested table visualization
        processed_results = self._preprocess_nested_tables(results)
        
        # Pre-process results to ensure clear visual distinction
        processed_results = self._preprocess_results_for_visual_clarity(processed_results)
        
        # Calculate summary statistics
        summary = self._generate_summary(processed_results)
        
        # Render HTML template
        html = self._render(processed_results, pdf1_name, pdf2_name, metadata or {}, summary)
        
        # Create unique filename
        filename = f"{self._safe(pdf1_name)}_vs_{self._safe(pdf2_name)}_{datetime.now():%Y%m%d_%H%M%S}.html"
        output_path = os.path.join(self.output_dir, filename)
        
        # Write the report to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
            
        logger.info(f"HTML report written to {output_path}")
        return output_path

    # ─── helpers ──────────────────────────────────────────────────
    def _build_table_registry(self, results: Dict):
        """Build a registry of all tables for cross-referencing."""
        self.table_registry = {}  # Reset registry
        for page_num, page in results.get("pages", {}).items():
            for table in page.get("table_differences", []):
                # Register using both table IDs if available
                if table.get("table_id1"):
                    self.table_registry[table.get("table_id1")] = table
                if table.get("table_id2"):
                    self.table_registry[table.get("table_id2")] = table
    
    def _preprocess_nested_tables(self, results: Dict) -> Dict:
        """Pre-process results to enhance nested table visualization."""
        processed = {**results}  # Create a copy
        
        # Define inner function for handling nested tables
        def add_nested_tables(parent_table, hierarchy, processed_tables):
            parent_id = parent_table.get("table_id1") or parent_table.get("table_id2")
            if not parent_id or parent_id not in hierarchy:
                return
                
            # Get children
            children_ids = hierarchy[parent_id]["children"]
            
            # Process each child
            for child_id in children_ids:
                if child_id in processed_tables:
                    continue
                    
                # Get child info
                child_info = hierarchy[child_id]
                child_table = child_info["table"]
                
                # Mark as processed
                processed_tables.add(child_id)
                
                # Add to parent's nested objects
                if "nested_table_objects" not in parent_table:
                    parent_table["nested_table_objects"] = []
                    
                parent_table["nested_table_objects"].append(child_table)
                
                # Process this child's children
                add_nested_tables(child_table, hierarchy, processed_tables)
        
        # Process nested relationships
        for page_num, page in processed.get("pages", {}).items():
            # Track processed tables to avoid duplicates
            processed_tables = set()
            
            # First pass: build the hierarchy tree
            table_hierarchy = {}
            for table in page.get("table_differences", []):
                table_id = table.get("table_id1") or table.get("table_id2")
                if not table_id:
                    continue
                    
                # Find parent ID
                parent_id = None
                for other_table in page.get("table_differences", []):
                    other_id = other_table.get("table_id1") or other_table.get("table_id2")
                    if other_id == table_id:
                        continue
                        
                    if other_table.get("has_nested_tables") and table_id in other_table.get("nested_tables", []):
                        parent_id = other_id
                        break
                
                # Store in hierarchy
                table_hierarchy[table_id] = {
                    "parent": parent_id,
                    "table": table,
                    "children": []
                }
            
            # Second pass: build child lists
            for table_id, info in table_hierarchy.items():
                if info["parent"]:
                    if info["parent"] in table_hierarchy:
                        table_hierarchy[info["parent"]]["children"].append(table_id)
            
            # Third pass: reorganize tables in processed results
            new_table_differences = []
            
            # Start with root tables (those without parents)
            for table_id, info in table_hierarchy.items():
                if not info["parent"] and table_id not in processed_tables:
                    # Add this table
                    new_table_differences.append(info["table"])
                    processed_tables.add(table_id)
                    
                    # Add nested tables property with full objects
                    info["table"]["nested_table_objects"] = []
                    
                    # Process children recursively using inner function
                    add_nested_tables(info["table"], table_hierarchy, processed_tables)
            
            # Replace table differences with new hierarchical version
            page["table_differences"] = new_table_differences
        
        return processed 
    def _add_nested_tables(self, parent_table, hierarchy, processed_tables):
            """Recursively add nested tables to parent table."""
            parent_id = parent_table.get("table_id1") or parent_table.get("table_id2")
            if not parent_id or parent_id not in hierarchy:
                return
                
            # Get children
            children_ids = hierarchy[parent_id]["children"]
            
            # Process each child
            for child_id in children_ids:
                if child_id in processed_tables:
                    continue
                    
                # Get child info
                child_info = hierarchy[child_id]
                child_table = child_info["table"]
                
                # Mark as processed
                processed_tables.add(child_id)
                
                # Add to parent's nested objects
                if "nested_table_objects" not in parent_table:
                    parent_table["nested_table_objects"] = []
                    
                parent_table["nested_table_objects"].append(child_table)
                
                # Process this child's children
                self._add_nested_tables(child_table, hierarchy, processed_tables)
                
    def _preprocess_results_for_visual_clarity(self, results: Dict) -> Dict:
        """
        Pre-process results to ensure clear visual distinction in the report.
        Adds visual_status to each element for consistent styling.
        """
        processed = {**results}  # Create a copy
        
        for page_num, page in processed.get("pages", {}).items():
            # Process text differences
            if "text_differences" in page:
                page["text_differences"] = self._standardize_text_differences(page["text_differences"])
            
            # Process table differences
            if "table_differences" in page:
                page["table_differences"] = self._standardize_table_differences(page["table_differences"])
        
        return processed

    def _standardize_text_differences(self, text_diffs):
        """Standardize text difference statuses for clear visual representation."""
        standardized = []
        
        for diff in text_diffs:
            status = diff.get("status", "")
            
            # Map statuses to standard visual categories
            if status == "deleted":
                visual_status = "deleted"  # RED
            elif status == "inserted":
                visual_status = "inserted"  # BLUE
            elif status in ["modified", "changed"]:
                visual_status = "modified"  # RED (for changed content)
            elif status == "similar":
                visual_status = "similar"  # GREEN
            elif status == "moved":
                visual_status = "moved"  # Special case
            elif status in ["equal", "matched"]:
                visual_status = "matched"  # No highlighting
            else:
                visual_status = status
            
            standardized_diff = diff.copy()
            standardized_diff["visual_status"] = visual_status
            standardized.append(standardized_diff)
        
        return standardized

    def _standardize_table_differences(self, table_diffs):
        """Standardize table difference statuses for clear visual representation."""
        standardized = []
        
        for diff in table_diffs:
            status = diff.get("status", "")
            
            # Map statuses to standard visual categories
            if status == "deleted":
                visual_status = "deleted"  # RED
            elif status == "inserted":
                visual_status = "inserted"  # BLUE
            elif status == "modified":
                visual_status = "modified"  # RED
            elif status == "moved":
                visual_status = "moved"  # Show as moved
            elif status == "matched":
                visual_status = "matched"  # No highlighting
            elif status == "similar":
                visual_status = "similar"  # GREEN
            else:
                visual_status = status
            
            standardized_diff = diff.copy()
            standardized_diff["visual_status"] = visual_status
            
            # Add visual_status to nested tables if present
            if "nested_table_objects" in standardized_diff:
                standardized_diff["nested_table_objects"] = self._standardize_table_differences(
                    standardized_diff["nested_table_objects"]
                )
            
            standardized.append(standardized_diff)
        
        return standardized            
    
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
        
        # Keep track of already counted nested tables
        counted_nested_tables = set()
        
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
                # Skip nested tables that have already been counted
                table_id = table.get("table_id1") or table.get("table_id2")
                if table_id in counted_nested_tables:
                    continue
                    
                # Process this table
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
                
                # Process nested tables if available
                if table.get("has_nested_tables") and "nested_table_objects" in table:
                    self._count_nested_tables(table.get("nested_table_objects", []), 
                                            counted_nested_tables, 
                                            matched_tables, moved_tables, modified_tables,
                                            deleted_tables, inserted_tables, 
                                            total_table, page_has_diff)
            
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

    def _count_nested_tables(self, nested_tables, counted_nested_tables, 
                        matched_tables, moved_tables, modified_tables,
                        deleted_tables, inserted_tables, total_table, page_has_diff):
        """Count nested tables for statistics."""
        for table in nested_tables:
            # Skip if already counted
            table_id = table.get("table_id1") or table.get("table_id2")
            if table_id in counted_nested_tables:
                continue
                
            # Mark as counted
            counted_nested_tables.add(table_id)
            
            # Count by status
            status = table.get("status", "")
            
            if status == "matched":
                matched_tables += 1
            elif status == "moved":
                moved_tables += 1
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
                
            # Recursively process nested tables
            if table.get("has_nested_tables") and "nested_table_objects" in table:
                self._count_nested_tables(table.get("nested_table_objects", []), 
                                        counted_nested_tables, 
                                        matched_tables, moved_tables, modified_tables,
                                        deleted_tables, inserted_tables, 
                                        total_table, page_has_diff)
                
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
        
    def _template_str(self):
        """Return the HTML template string for the comparison report."""
        return """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                
                header {
                    background-color: #4285F4;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                }
                
                h1, h2, h3 {
                    margin-top: 0;
                }
                
                .summary {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }
                
                .stat-item {
                    background-color: #f8f9fa;
                    border-radius: 4px;
                    padding: 15px;
                    text-align: center;
                }
                
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4285F4;
                    margin: 5px 0;
                }
                
                .stat-label {
                    font-size: 14px;
                    color: #666;
                }
                
                .page-nav {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 5px;
                    margin-bottom: 20px;
                }
                
                .page-button {
                    padding: 8px 12px;
                    background-color: #f1f1f1;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                    color: #333;
                }
                
                .page-button:hover {
                    background-color: #e0e0e0;
                }
                
                .page-button.active {
                    background-color: #4285F4;
                    color: white;
                    border-color: #4285F4;
                }
                
                .page-section {
                    margin-bottom: 30px;
                    border: 1px solid #eee;
                    border-radius: 4px;
                    padding: 20px;
                }
                
                .page-header {
                    background-color: #f8f9fa;
                    padding: 10px;
                    margin: -20px -20px 20px -20px;
                    border-bottom: 1px solid #eee;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .page-title {
                    margin: 0;
                    color: #4285F4;
                }
                
                .differences-section {
                    margin-top: 15px;
                }
                
                .difference-item {
                    margin-bottom: 20px;
                    border: 1px solid #eee;
                    border-radius: 4px;
                    overflow: hidden;
                }
                
                .difference-header {
                    padding: 10px 15px;
                    background-color: #f8f9fa;
                    border-bottom: 1px solid #eee;
                    font-weight: bold;
                }
                
                .difference-content {
                    padding: 15px;
                }
                
                /* Text difference styles */
                .text-difference {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }
                
                .text-source1, .text-source2 {
                    padding: 10px;
                    border: 1px solid #eee;
                    border-radius: 4px;
                    background-color: #f9f9f9;
                }
                
                /* Table difference styles */
                .table-diff-container {
                    display: flex;
                    width: 100%;
                    overflow-x: auto;
                }
                
                .table-diff-left, .table-diff-right {
                    flex: 1;
                    padding: 10px;
                    min-width: 300px;
                }
                
                .diff-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                }
                
                .diff-table th, .diff-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                
                .diff-table th {
                    background-color: #f2f2f2;
                    position: sticky;
                    top: 0;
                }
                
                .diff-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                
                .row-index {
                    font-weight: bold;
                    text-align: center;
                    background-color: #f2f2f2;
                }
                
                /* Cell status styles */
                .similar { 
                    background-color: #EDFFF0; 
                    color: #00AA00; 
                }
                
                .modified { 
                    background-color: #FCF8E3; 
                    color: #EA4335; 
                }
                
                .deleted { 
                    background-color: #FFEDED; 
                    color: #FF0000; 
                }
                
                .inserted { 
                    background-color: #EDF5FF; 
                    color: #0000FF; 
                }
                
                .empty {
                    background-color: #f9f9f9;
                    color: #999;
                }
                
                /* Status indicators */
                .status-badge {
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    text-transform: uppercase;
                }
                
                .status-matched {
                    background-color: #D4EDDA;
                    color: #155724;
                }
                
                .status-modified {
                    background-color: #FCF8E3;
                    color: #856404;
                }
                
                .status-deleted {
                    background-color: #F8D7DA;
                    color: #721C24;
                }
                
                .status-inserted {
                    background-color: #CCE5FF;
                    color: #004085;
                }
                
                .status-moved {
                    background-color: #D1ECF1;
                    color: #0C5460;
                }
                
                /* Nested table styles */
                .nested-tables {
                    margin-top: 15px;
                    padding-left: 20px;
                    border-left: 3px solid #4285F4;
                }
                
                .metadata-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                
                .metadata-table th, .metadata-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                
                .metadata-table th {
                    background-color: #f2f2f2;
                    width: 200px;
                }
                
                /* Legend */
                .diff-legend {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    padding: 15px;
                    background-color: #f8f8f8;
                    border-radius: 5px;
                    margin: 15px 0;
                }
                
                .legend-item {
                    display: flex;
                    align-items: center;
                    font-size: 0.9rem;
                }
                
                .legend-color {
                    width: 16px;
                    height: 16px;
                    border-radius: 3px;
                    margin-right: 6px;
                }
                
                /* Table of contents */
                .toc {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }
                
                .toc-title {
                    margin-top: 0;
                    margin-bottom: 10px;
                }
                
                .toc-list {
                    list-style-type: none;
                    padding-left: 0;
                }
                
                .toc-item {
                    margin-bottom: 5px;
                }
                
                .toc-link {
                    text-decoration: none;
                    color: #4285F4;
                }
                
                .toc-link:hover {
                    text-decoration: underline;
                }
                
                /* Page anchors */
                .page-anchor {
                    display: block;
                    position: relative;
                    top: -50px;
                    visibility: hidden;
                }
                
                /* Responsive adjustments */
                @media (max-width: 768px) {
                    .summary {
                        grid-template-columns: 1fr 1fr;
                    }
                    
                    .text-difference {
                        grid-template-columns: 1fr;
                    }
                    
                    .table-diff-container {
                        flex-direction: column;
                    }
                    
                    .table-diff-left, .table-diff-right {
                        min-width: 100%;
                    }
                }
                
                @media (max-width: 480px) {
                    .summary {
                        grid-template-columns: 1fr;
                    }
                    
                    .page-header {
                        flex-direction: column;
                        align-items: flex-start;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <header>
                    <h1>{{ title }}</h1>
                    <p>Comparison Date: {{ comparison_date }}</p>
                </header>
                
                <div class="metadata">
                    <table class="metadata-table">
                        <tr>
                            <th>First PDF</th>
                            <td>{{ pdf1_name }}</td>
                        </tr>
                        <tr>
                            <th>Second PDF</th>
                            <td>{{ pdf2_name }}</td>
                        </tr>
                        {% for key, value in metadata.items() %}
                        <tr>
                            <th>{{ key }}</th>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                <div class="summary">
                    <div class="stat-item">
                        <div class="stat-label">Total Pages</div>
                        <div class="stat-value">{{ summary.total_pages }}</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">Pages with Differences</div>
                        <div class="stat-value">{{ summary.pages_with_differences }}</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">Text Differences</div>
                        <div class="stat-value">{{ summary.total_text_differences }}</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">Table Differences</div>
                        <div class="stat-value">{{ summary.total_table_differences }}</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">Total Differences</div>
                        <div class="stat-value">{{ summary.total_differences }}</div>
                    </div>
                    
                    <div class="stat-item">
                        <div class="stat-label">Changed Pages %</div>
                        <div class="stat-value">{{ summary.percentage_changed_pages }}%</div>
                    </div>
                </div>
                
                <!-- Color Legend -->
                <div class="diff-legend">
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FFEDED; border: 1px solid #FF0000;"></div>
                        <span>Deleted Content (Red)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #EDF5FF; border: 1px solid #0000FF;"></div>
                        <span>New Content (Blue)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #FCF8E3; border: 1px solid #EA4335;"></div>
                        <span>Modified Content (Red)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #EDFFF0; border: 1px solid #00AA00;"></div>
                        <span>Similar Content (Green)</span>
                    </div>
                </div>
                
                <!-- Table of Contents -->
                <div class="toc">
                    <h3 class="toc-title">Table of Contents</h3>
                    <ul class="toc-list">
                        {% for page_num in range(1, results.max_pages + 1) %}
                        <li class="toc-item">
                            <a href="#page-{{ page_num }}" class="toc-link">Page {{ page_num }}</a>
                        </li>
                        {% endfor %}
                    </ul>
                </div>
                
                <!-- Page Navigation -->
                <div class="page-nav">
                    {% for page_num in range(1, results.max_pages + 1) %}
                    <a href="#page-{{ page_num }}" class="page-button">{{ page_num }}</a>
                    {% endfor %}
                </div>
                
                <!-- Pages Content -->
                {% for page_num in range(1, results.max_pages + 1) %}
                <a id="page-{{ page_num }}" class="page-anchor"></a>
                <div class="page-section">
                    <div class="page-header">
                        <h2 class="page-title">Page {{ page_num }}</h2>
                        
                        {% if page_num in results.pages and (results.pages[page_num].text_differences|length > 0 or results.pages[page_num].table_differences|length > 0) %}
                        <span class="status-badge status-modified">Has Differences</span>
                        {% else %}
                        <span class="status-badge status-matched">Identical</span>
                        {% endif %}
                    </div>
                    
                    {% if page_num in results.pages %}
                    
                    <!-- Text Differences -->
                    {% if results.pages[page_num].text_differences|length > 0 %}
                    <div class="differences-section">
                        <h3>Text Differences ({{ results.pages[page_num].text_differences|length }})</h3>
                        
                        {% for diff in results.pages[page_num].text_differences %}
                        <div class="difference-item">
                            <div class="difference-header">
                                <span class="status-badge status-{{ diff.visual_status }}">{{ diff.status }}</span>
                                {% if diff.score %}
                                <span style="margin-left: 10px;">Similarity: {{ "%.2f"|format(diff.score * 100) }}%</span>
                                {% endif %}
                            </div>
                            
                            <div class="difference-content">
                                <div class="text-difference">
                                    <div class="text-source1">
                                        <strong>PDF 1:</strong>
                                        <div>{{ diff.text1 }}</div>
                                    </div>
                                    
                                    <div class="text-source2">
                                        <strong>PDF 2:</strong>
                                        <div>{{ diff.text2 }}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    <!-- Table Differences -->
                    {% if results.pages[page_num].table_differences|length > 0 %}
                    <div class="differences-section">
                        <h3>Table Differences ({{ results.pages[page_num].table_differences|length }})</h3>
                        
                        {% for diff in results.pages[page_num].table_differences %}
                        <div class="difference-item">
                            <div class="difference-header">
                                <span class="status-badge status-{{ diff.visual_status }}">{{ diff.status }}</span>
                                
                                {% if diff.similarity %}
                                <span style="margin-left: 10px;">Similarity: {{ "%.2f"|format(diff.similarity * 100) }}%</span>
                                {% endif %}
                                
                                {% if diff.differences %}
                                <span style="margin-left: 10px;">Differences: {{ diff.differences }}</span>
                                {% endif %}
                            </div>
                            
                            <div class="difference-content">
                                {% if diff.diff_html %}
                                {{ diff.diff_html|safe }}
                                {% else %}
                                <p>Tables are identical or have been moved without changes.</p>
                                {% endif %}
                                
                                <!-- Nested Tables -->
                                {% if diff.nested_table_objects and diff.nested_table_objects|length > 0 %}
                                <div class="nested-tables">
                                    <h4>Nested Tables ({{ diff.nested_table_objects|length }})</h4>
                                    
                                    {% for nested in diff.nested_table_objects %}
                                    <div class="difference-item">
                                        <div class="difference-header">
                                            <span class="status-badge status-{{ nested.visual_status }}">{{ nested.status }}</span>
                                            
                                            {% if nested.similarity %}
                                            <span style="margin-left: 10px;">Similarity: {{ "%.2f"|format(nested.similarity * 100) }}%</span>
                                            {% endif %}
                                            
                                            {% if nested.differences %}
                                            <span style="margin-left: 10px;">Differences: {{ nested.differences }}</span>
                                            {% endif %}
                                        </div>
                                        
                                        <div class="difference-content">
                                            {% if nested.diff_html %}
                                            {{ nested.diff_html|safe }}
                                            {% else %}
                                            <p>Nested tables are identical or have been moved without changes.</p>
                                            {% endif %}
                                        </div>
                                    </div>
                                    {% endfor %}
                                </div>
                                {% endif %}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                    {% endif %}
                    
                    {% else %}
                    <p>No differences found on this page.</p>
                    {% endif %}
                </div>
                {% endfor %}
                
            </div>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    // Highlight current page in navigation
                    const pageButtons = document.querySelectorAll('.page-button');
                    
                    function updateActiveButton() {
                        const scrollPos = window.scrollY;
                        const pageAnchors = document.querySelectorAll('.page-anchor');
                        
                        pageAnchors.forEach((anchor, i) => {
                            const rect = anchor.getBoundingClientRect();
                            if (rect.top + window.scrollY - 100 <= scrollPos && (i === pageAnchors.length - 1 || 
                                pageAnchors[i+1].getBoundingClientRect().top + window.scrollY - 100 > scrollPos)) {
                                pageButtons.forEach(btn => btn.classList.remove('active'));
                                pageButtons[i].classList.add('active');
                            }
                        });
                    }
                    
                    // Initial update
                    updateActiveButton();
                    
                    // Update on scroll
                    window.addEventListener('scroll', updateActiveButton);
                });
            </script>
        </body>
        </html>
        """      
        
    @staticmethod
    def _safe(name):
        """Create a safe filename from the document name."""
        return re.sub(r"[^\w\-\.]", "_", name)[:50]    
