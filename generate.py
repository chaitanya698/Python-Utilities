"""
generate.py - Enhanced report builder
------------------
Improvements:
* Better visualization of nested tables
* Enhanced side-by-side diff for tables
* Color-coded differences (modified: red, new: blue, similar: green)
* Improved summary statistics
* Interactive UI elements
* Support for multi-page tables
* Clearer visual distinction for differences
* Side-by-side comparison of both PDFs
"""
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
        
        # Build table registry for nested table visualization
        self._build_table_registry(results)
        
        # Pre-process results to enhance nested table visualization
        processed_results = self._preprocess_nested_tables(results)
        
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
                    
                    # Process children recursively
                    self._add_nested_tables(info["table"], table_hierarchy, processed_tables)
            
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
                
                # Process nested tables
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

    @staticmethod
    def _safe(name):
        """Create a safe filename from the document name."""
        return re.sub(r"[^\w\-\.]", "_", name)[:50]

    # ─── HTML template ────────────────────────────────────────────
    @staticmethod
    def _template_str() -> str:
        """Return the enhanced HTML template for the comparison report."""
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
            
            /* Specific colors for differences */
            --deleted-bg: #FFEDED;
            --deleted-color: #FF0000;
            --inserted-bg: #EDF5FF;
            --inserted-color: #0000FF;
            --modified-bg: #FCF8E3;
            --modified-color: #EA4335;
            --similar-bg: #EDFFF0;
            --similar-color: #00AA00;
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
            max-width: 100%;
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
            background-color: var(--danger-color);
            color: white;
        }
        
        .page-badge.no-diff {
            background-color: var(--success-color);
            color: white;
        }
        
        /* Side-by-side layout */
        .side-by-side-container {
            display: flex;
            width: 100%;
            overflow: hidden;
            margin-bottom: 20px;
            border: 1px solid #e5e5e5;
            border-radius: 5px;
        }
        
        .pdf-column {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            max-height: calc(100vh - 200px);
            border-right: 1px solid #e5e5e5;
        }
        
        .pdf-column:last-child {
            border-right: none;
        }
        
        .pdf-column-header {
            background-color: #f5f5f5;
            padding: 10px;
            font-weight: bold;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
            margin: -15px -15px 15px -15px;
            position: sticky;
            top: 0;
            z-index: 10;
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
            color: var(--deleted-color);
            border-left: 4px solid var(--deleted-color);
        }
        
        .diff-header.inserted {
            background-color: var(--inserted-bg);
            color: var(--inserted-color);
            border-left: 4px solid var(--inserted-color);
        }
        
        .diff-header.modified {
            background-color: var(--modified-bg);
            color: var(--modified-color);
            border-left: 4px solid var(--modified-color);
        }
        
        .diff-header.similar {
            background-color: var(--similar-bg);
            color: var(--similar-color);
            border-left: 4px solid var(--similar-color);
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
            color: var(--deleted-color);
        }
        
        .text-diff-line.inserted {
            background-color: var(--inserted-bg);
            color: var(--inserted-color);
        }
        
        .text-diff-line.changed,
        .text-diff-line.modified {
            background-color: var(--modified-bg);
            color: var(--modified-color);
        }
        
        .text-diff-line.similar {
            background-color: var(--similar-bg);
            color: var(--similar-color);
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
            color: var(--deleted-color);
            padding: 0 2px;
            border-radius: 2px;
            font-weight: bold;
        }
        
        .diff-inserted {
            background-color: var(--inserted-bg);
            color: var(--inserted-color);
            padding: 0 2px;
            border-radius: 2px;
            font-weight: bold;
        }
        
        .diff-similar {
            background-color: var(--similar-bg);
            color: var(--similar-color);
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
        
        /* Enhanced table cell styles */
        .diff-table.deleted {
            border: 2px solid var(--deleted-color);
        }
        
        .diff-table.inserted {
            border: 2px solid var(--inserted-color);
        }
        
        .diff-table td.deleted {
            background-color: var(--deleted-bg);
            color: var(--deleted-color);
            font-weight: bold;
        }
        
        .diff-table td.inserted {
            background-color: var(--inserted-bg);
            color: var(--inserted-color);
            font-weight: bold;
        }
        
        .diff-table td.modified {
            background-color: var(--modified-bg);
            color: var(--modified-color);
            font-weight: bold;
        }
        
        .diff-table td.similar {
            background-color: var(--similar-bg);
            color: var(--similar-color);
        }
        
        /* Enhanced table styles with cell coordinates */
        .column-headers th {
            background-color: #f0f0f0;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        
        .row-index {
            background-color: #f0f0f0;
            font-weight: bold;
            position: sticky;
            left: 0;
            z-index: 5;
        }
        
        .row-index-header {
            background-color: #e0e0e0;
            position: sticky;
            top: 0;
            left: 0;
            z-index: 15;
        }
        
        /* Table cell highlighting on hover */
        .highlight-cell {
            background-color: #e0f7fa !important;
            position: relative;
            z-index: 1;
        }
        
        .highlight-row td {
            background-color: rgba(224, 247, 250, 0.5);
        }
        
        .highlight-col {
            background-color: rgba(224, 247, 250, 0.5);
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
            margin-top: 10px;
            margin-bottom: 10px;
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
        
        /* Navigation elements */
        .toc {
            background: #f8f8f8;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .toc-title {
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .toc-links {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .toc-link {
            display: inline-block;
            padding: 3px 8px;
            background: #e9e9e9;
            border-radius: 4px;
            color: var(--dark-color);
            text-decoration: none;
            font-size: 0.9rem;
        }
        
        .toc-link:hover {
            background: #d9d9d9;
        }
        
        .toc-link.has-diff {
            background-color: var(--danger-color);
            color: white;
        }
        
        .back-to-top {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: var(--primary-color);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s;
            opacity: 0;
            pointer-events: none;
            z-index: 1000;
        }
        
        .back-to-top.visible {
            opacity: 1;
            pointer-events: auto;
        }
        
        .back-to-top:hover {
            background: #3367d6;
            transform: translateY(-3px);
        }
        
        /* Legend for diff colors */
        .diff-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 0.9rem;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            margin-right: 5px;
        }
        
        .legend-deleted {
            background-color: var(--deleted-bg);
            border: 1px solid var(--deleted-color);
        }
        
        .legend-inserted {
            background-color: var(--inserted-bg);
            border: 1px solid var(--inserted-color);
        }
        
        .legend-modified {
            background-color: var(--modified-bg);
            border: 1px solid var(--modified-color);
        }
        
        .legend-similar {
            background-color: var(--similar-bg);
            border: 1px solid var(--similar-color);
        }
        
        /* Responsive design */
        @media (max-width: 992px) {
            .side-by-side-container {
                flex-direction: column;
            }
            
            .pdf-column {
                border-right: none;
                border-bottom: 1px solid #e5e5e5;
                max-height: none;
            }
        }
        
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
            
            .inline-diff {
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
            
            <!-- Color Legend -->
            <div class="diff-legend">
                <div class="legend-item">
                    <div class="legend-color legend-deleted"></div>
                    <span>Deleted Content (Red) - only in {{ pdf1_name }}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-inserted"></div>
                    <span>Inserted Content (Blue) - only in {{ pdf2_name }}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-modified"></div>
                    <span>Modified Content (Red) - changed between versions</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color legend-similar"></div>
                    <span>Similar Content (Green) - semantically equivalent</span>
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
        
        <!-- Table of Contents -->
        <section class="toc">
            <div class="toc-title">Quick Navigation</div>
            <div class="toc-links">
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
                    
                    <a href="#page-{{ page_num }}" class="toc-link page-link {{ 'has-diff' if has_differences else '' }}">
                        Page {{ page_num }}
                    </a>
                {% endfor %}
            </div>
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
                
                <!-- Side-by-side container for this page -->
                <div class="side-by-side-container">
                    <!-- Left column (PDF 1) -->
                    <div class="pdf-column">
                        <div class="pdf-column-header">{{ pdf1_name }}</div>
                        
                        <!-- Text content from PDF 1 -->
                        {% set has_pdf1_content = false %}
                        
                        {% if page.text_differences %}
                            <div class="text-differences">
                                <div class="diff-header">Text Content</div>
                                {% for diff in page.text_differences %}
                                    {% if diff.status == 'equal' %}
                                        <div class="text-diff-line similar">
                                            <span class="line-number">{{ diff.line_num1 + 1 if diff.line_num1 is not none else '-' }}</span>
                                            <span class="text-content">{{ diff.text1 }}</span>
                                        </div>
                                        {% set has_pdf1_content = true %}
                                    {% elif diff.status == 'deleted' %}
                                        <div class="text-diff-line deleted">
                                            <span class="line-number">{{ diff.line_num1 + 1 }}</span>
                                            <span class="text-content">{{ diff.text1 }}</span>
                                        </div>
                                        {% set has_pdf1_content = true %}
                                    {% elif diff.status in ['changed', 'modified'] and diff.line_num1 is not none %}
                                        <div class="text-diff-line modified">
                                            <span class="line-number">{{ diff.line_num1 + 1 }}</span>
                                            <span class="text-content">{{ diff.text1 }}</span>
                                        </div>
                                        {% set has_pdf1_content = true %}
                                    {% endif %}
                                {% endfor %}
                            </div>
                        {% endif %}
                        
                        <!-- Tables from PDF 1 -->
                        {% for t in page.table_differences %}
                            {% if t.status in ['matched', 'modified', 'deleted', 'moved'] %}
                                {% if t.status == 'matched' %}
                                    <div class="diff-header similar">
                                        Table {{ t.table_id1 }} (Identical in both PDFs)
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display table content from PDF 1 -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf1_content = true %}
                                {% elif t.status == 'moved' %}
                                    <div class="diff-header similar">
                                        Table {{ t.table_id1 }}
                                        <span class="moved-badge">Moved to page {{ t.page2 }} in {{ pdf2_name }}</span>
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display table content from PDF 1 -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf1_content = true %}
                                {% elif t.status == 'modified' %}
                                    <div class="diff-header modified">
                                        Table {{ t.table_id1 }} (Modified - {{ t.differences }} differences)
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- For modified table, show the left side of diff -->
                                    {% if t.diff_html %}
                                        {{ t.diff_html|safe }}
                                    {% else %}
                                        <div>Table content not available</div>
                                    {% endif %}
                                    {% set has_pdf1_content = true %}
                                {% elif t.status == 'deleted' %}
                                    <div class="diff-header deleted">
                                        Table {{ t.table_id1 }} (Only in {{ pdf1_name }})
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display deleted table -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf1_content = true %}
                                {% endif %}
                                
                                <!-- Display nested tables if any -->
                                {% if t.nested_table_objects %}
                                    <div class="nested-table-container">
                                        {% for nested in t.nested_table_objects %}
                                            {% if nested.status == 'matched' %}
                                                <div class="diff-header similar">
                                                    Nested Table {{ nested.table_id1 }} (Identical)
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'moved' %}
                                                <div class="diff-header similar">
                                                    Nested Table {{ nested.table_id1 }}
                                                    <span class="moved-badge">Moved from page {{ nested.page1 }} to {{ nested.page2 }}</span>
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'modified' and nested.table_id1 %}
                                                <div class="diff-header modified">
                                                    Nested Table {{ nested.table_id1 }} (Modified - {{ nested.differences }} differences)
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'deleted' %}
                                                <div class="diff-header deleted">
                                                    Nested Table (Only in {{ pdf1_name }})
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% endif %}
                                        {% endfor %}
                                    </div>
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                        
                        {% if not has_pdf1_content %}
                            <div class="diff-header">No content on this page</div>
                        {% endif %}
                    </div>
                    
                    <!-- Right column (PDF 2) -->
                    <div class="pdf-column">
                        <div class="pdf-column-header">{{ pdf2_name }}</div>
                        
                        <!-- Text content from PDF 2 -->
                        {% set has_pdf2_content = false %}
                        
                        {% if page.text_differences %}
                            <div class="text-differences">
                                <div class="diff-header">Text Content</div>
                                {% for diff in page.text_differences %}
                                    {% if diff.status == 'equal' %}
                                        <div class="text-diff-line similar">
                                            <span class="line-number">{{ diff.line_num2 + 1 if diff.line_num2 is not none else '-' }}</span>
                                            <span class="text-content">{{ diff.text2 }}</span>
                                        </div>
                                        {% set has_pdf2_content = true %}
                                    {% elif diff.status == 'inserted' %}
                                        <div class="text-diff-line inserted">
                                            <span class="line-number">{{ diff.line_num2 + 1 }}</span>
                                            <span class="text-content">{{ diff.text2 }}</span>
                                        </div>
                                        {% set has_pdf2_content = true %}
                                    {% elif diff.status in ['changed', 'modified'] and diff.line_num2 is not none %}
                                        <div class="text-diff-line modified">
                                            <span class="line-number">{{ diff.line_num2 + 1 }}</span>
                                            <span class="text-content">{{ diff.text2 }}</span>
                                        </div>
                                        {% set has_pdf2_content = true %}
                                    {% endif %}
                                {% endfor %}
                            </div>
                        {% endif %}
                        
                        <!-- Tables from PDF 2 -->
                        {% for t in page.table_differences %}
                            {% if t.status in ['matched', 'modified', 'inserted', 'moved'] %}
                                {% if t.status == 'matched' %}
                                    <div class="diff-header similar">
                                        Table {{ t.table_id2 }} (Identical in both PDFs)
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display table content from PDF 2 -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf2_content = true %}
                                {% elif t.status == 'moved' and t.page2 == page_num %}
                                    <div class="diff-header similar">
                                        Table {{ t.table_id2 }}
                                        <span class="moved-badge">Moved from page {{ t.page1 }} in {{ pdf1_name }}</span>
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display table content from PDF 2 -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf2_content = true %}
                                {% elif t.status == 'modified' %}
                                    <div class="diff-header modified">
                                        Table {{ t.table_id2 }} (Modified - {{ t.differences }} differences)
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- For modified table, show the right side of diff -->
                                    {% if t.diff_html %}
                                        {{ t.diff_html|safe }}
                                    {% else %}
                                        <div>Table content not available</div>
                                    {% endif %}
                                    {% set has_pdf2_content = true %}
                                {% elif t.status == 'inserted' %}
                                    <div class="diff-header inserted">
                                        Table {{ t.table_id2 }} (Only in {{ pdf2_name }})
                                        {% if t.has_nested_tables %}
                                            <span class="nested-table-indicator">Contains nested tables</span>
                                        {% endif %}
                                    </div>
                                    <!-- Display inserted table -->
                                    {{ t.diff_html|safe if t.diff_html else '<div>Table content not available</div>' }}
                                    {% set has_pdf2_content = true %}
                                {% endif %}
                                
                                <!-- Display nested tables if any -->
                                {% if t.nested_table_objects %}
                                    <div class="nested-table-container">
                                        {% for nested in t.nested_table_objects %}
                                            {% if nested.status == 'matched' %}
                                                <div class="diff-header similar">
                                                    Nested Table {{ nested.table_id2 }} (Identical)
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'moved' and nested.page2 == page_num %}
                                                <div class="diff-header similar">
                                                    Nested Table {{ nested.table_id2 }}
                                                    <span class="moved-badge">Moved from page {{ nested.page1 }} to {{ nested.page2 }}</span>
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'modified' and nested.table_id2 %}
                                                <div class="diff-header modified">
                                                    Nested Table {{ nested.table_id2 }} (Modified - {{ nested.differences }} differences)
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% elif nested.status == 'inserted' %}
                                                <div class="diff-header inserted">
                                                    Nested Table (Only in {{ pdf2_name }})
                                                </div>
                                                {{ nested.diff_html|safe if nested.diff_html else '<div>Nested table content not available</div>' }}
                                            {% endif %}
                                        {% endfor %}
                                    </div>
                                {% endif %}
                            {% endif %}
                        {% endfor %}
                        
                        {% if not has_pdf2_content %}
                            <div class="diff-header">No content on this page</div>
                        {% endif %}
                    </div>
                </div>
            </section>
        {% endfor %}
        
        <footer>
            Report generated on {{ comparison_date }} | PDF Comparison Tool v2.0
        </footer>
    </div>
    
    <a href="#" class="back-to-top" id="back-to-top">↑</a>
    
    <script>
        // Initialize UI enhancements when the document is ready
        document.addEventListener('DOMContentLoaded', function() {
            // Make diff headers collapsible
            const collapsibles = document.querySelectorAll('.collapsible');
            collapsibles.forEach(function(item) {
                item.addEventListener('click', function() {
                    this.classList.toggle('expanded');
                    
                    // Toggle content visibility
                    const content = this.nextElementSibling;
                    if (content.classList.contains('collapsible-content')) {
                        content.classList.toggle('expanded');
                    }
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
            
            // Back to top button
            const backToTop = document.getElementById('back-to-top');
            
            // Show/hide back to top button
            window.addEventListener('scroll', function() {
                if (window.pageYOffset > 300) {
                    backToTop.classList.add('visible');
                } else {
                    backToTop.classList.remove('visible');
                }
            });
            
            // Scroll to top when clicking the button
            backToTop.addEventListener('click', function(e) {
                e.preventDefault();
                window.scrollTo({
                    top: 0,
                    behavior: 'smooth'
                });
            });
            
            // Enhanced table cell interaction
            const tableCells = document.querySelectorAll('.diff-table td[data-col]');
            tableCells.forEach(function(cell) {
                cell.addEventListener('mouseenter', function() {
                    // Get cell coordinates
                    const col = this.getAttribute('data-col');
                    const row = this.parentElement.getAttribute('data-row');
                    
                    // Find the container this cell belongs to
                    const container = this.closest('.side-by-side-container');
                    if (!container) return;
                    
                    // Select cells in both tables with matching coordinates
                    const sameColCells = container.querySelectorAll(`.diff-table td[data-col="${col}"]`);
                    sameColCells.forEach(c => c.classList.add('highlight-col'));
                    
                    const sameRowCells = container.querySelectorAll(`.diff-table tr[data-row="${row}"] td`);
                    sameRowCells.forEach(c => c.classList.add('highlight-row'));
                    
                    // Extra highlight for this specific cell
                    this.classList.add('highlight-cell');
                });
                
                cell.addEventListener('mouseleave', function() {
                    // Remove all highlighting
                    document.querySelectorAll('.highlight-col, .highlight-row, .highlight-cell').forEach(el => {
                        el.classList.remove('highlight-col');
                        el.classList.remove('highlight-row');
                        el.classList.remove('highlight-cell');
                    });
                });
            });
            
            // Synchronize scrolling between side-by-side columns
            const pdfColumns = document.querySelectorAll('.pdf-column');
            
            // Group columns by their container
            const columnPairs = {};
            document.querySelectorAll('.side-by-side-container').forEach((container, index) => {
                const columns = container.querySelectorAll('.pdf-column');
                if (columns.length === 2) {
                    columnPairs[index] = {
                        left: columns[0],
                        right: columns[1],
                        syncing: false // Flag to prevent recursive scroll events
                    };
                }
            });
            
            // Add scroll event listeners to each column
            Object.values(columnPairs).forEach(pair => {
                pair.left.addEventListener('scroll', function() {
                    if (!pair.syncing) {
                        pair.syncing = true;
                        // Get scroll position as percentage
                        const scrollPercent = this.scrollTop / (this.scrollHeight - this.clientHeight);
                        // Apply to right column
                        pair.right.scrollTop = scrollPercent * (pair.right.scrollHeight - pair.right.clientHeight);
                        setTimeout(() => { pair.syncing = false; }, 50);
                    }
                });
                
                pair.right.addEventListener('scroll', function() {
                    if (!pair.syncing) {
                        pair.syncing = true;
                        // Get scroll position as percentage
                        const scrollPercent = this.scrollTop / (this.scrollHeight - this.clientHeight);
                        // Apply to left column
                        pair.left.scrollTop = scrollPercent * (pair.left.scrollHeight - pair.left.clientHeight);
                        setTimeout(() => { pair.syncing = false; }, 50);
                    }
                });
            });
        });
    </script>
</body>
</html>
"""
