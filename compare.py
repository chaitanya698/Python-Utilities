"""
compare.py
----------
Enhanced diff engine for text & table comparison.

Key improvements:
* Content-based matching for tables regardless of page position
* Support for multi-page table comparison
* Improved diff visualization with column-level highlighting
* Detailed cell-level difference detection
"""

import logging
import re
import itertools
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PdfCompare:
    def __init__(self, 
                 diff_threshold: float = 0.75,
                 cell_match_threshold: float = 0.9,
                 fuzzy_match_threshold: float = 0.8):
        self.diff_threshold = diff_threshold        # Table similarity threshold
        self.cell_match_threshold = cell_match_threshold  # Cell content similarity threshold  
        self.fuzzy_match_threshold = fuzzy_match_threshold  # Fuzzy text matching threshold

    # ─── public ───────────────────────────────────────────────────
    def compare_pdfs(self, pdf1: Dict, pdf2: Dict) -> Dict:
        """
        Compare two PDFs and generate detailed differences.
        
        Args:
            pdf1: First PDF structure from extractor
            pdf2: Second PDF structure from extractor
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting PDF comparison")
        max_pages = max(len(pdf1), len(pdf2))
        results = {"max_pages": max_pages, "pages": {}}

        # Pre-process: collect all tables from both PDFs
        tables1 = list(self._collect_tables(pdf1))
        tables2 = list(self._collect_tables(pdf2))
        
        logger.info(f"Found {len(tables1)} tables in first PDF and {len(tables2)} tables in second PDF")
        
        # Match tables across both PDFs using content-based matching
        table_matches = self._match_tables_global(tables1, tables2)
        
        # Organize table matches by page for the report
        tables_by_page = defaultdict(list)
        for diff in table_matches:
            for pg in (diff.get("page1"), diff.get("page2")):
                if pg is not None:
                    tables_by_page[pg].append(diff)

        # Process each page
        for p in range(1, max_pages + 1):
            logger.info(f"Processing page {p}/{max_pages}")
            
            # Get text elements from both PDFs
            txt1 = pdf1.get(p, {"elements": []}).get("elements", [])
            txt2 = pdf2.get(p, {"elements": []}).get("elements", [])
            
            # Filter for text-only elements
            text1 = [e for e in txt1 if e["type"] == "text"]
            text2 = [e for e in txt2 if e["type"] == "text"]
            
            # Perform text comparison
            text_diff = self._compare_text_elements(text1, text2)
            
            # Get table differences for this page
            table_diff = tables_by_page.get(p, [])
            
            # Store results for this page
            results["pages"][p] = {
                "text_differences": text_diff,
                "table_differences": table_diff
            }

        logger.info("PDF comparison completed")
        return results

    # ─── table processing ─────────────────────────────────────────
    def _collect_tables(self, pdf_data):
        """Extract all tables from PDF data."""
        for pg, pdata in pdf_data.items():
            for el in pdata.get("elements", []):
                if el.get("type") == "table":
                    yield (pg, el)

    def _match_tables_global(self, tables1, tables2) -> List[Dict]:
        """
        Match tables between two PDFs using content-based similarity.
        
        This improved algorithm:
        1. Prioritizes content matching over position matching
        2. Handles multi-page tables correctly
        3. Provides detailed cell-level diff information
        """
        results = []
        matched_tables2 = set()
        
        # If either PDF has no tables
        if not tables1 or not tables2:
            # Handle case where tables exist in only one PDF
            if tables1:
                results.extend([self._build_table_diff("deleted", pg, None, t, None) 
                               for pg, t in tables1])
            if tables2:
                results.extend([self._build_table_diff("inserted", None, pg, None, t) 
                               for pg, t in tables2])
            return results
        
        # First, try to match tables using content hash (exact content match)
        content_matches = self._match_by_content_hash(tables1, tables2)
        
        # Process content matches first (exact matches)
        for (pg1, t1), (pg2, t2) in content_matches:
            status = "moved" if pg1 != pg2 else "matched"
            results.append(self._build_table_diff(
                status, pg1, pg2, t1, t2, 
                differences=0, 
                similarity=1.0
            ))
            matched_tables2.add((pg2, t2["table_id"]))
        
        # For remaining tables, use similarity-based matching
        remaining1 = [(pg, t) for pg, t in tables1 
                     if not any((pg, t) == match[0] for match in content_matches)]
        remaining2 = [(pg, t) for pg, t in tables2 
                     if (pg, t["table_id"]) not in matched_tables2]
        
        # For each remaining table in first PDF, find best match in second PDF
        for pg1, t1 in remaining1:
            best_match = None
            best_sim = 0.0
            best_diffs = 0
            best_diff_html = ""
            
            for pg2, t2 in remaining2:
                if (pg2, t2["table_id"]) in matched_tables2:
                    continue
                    
                # Calculate similarity and differences
                sim, diffs, diff_html = self._compare_tables(t1["content"], t2["content"])
                
                if sim > best_sim:
                    best_sim = sim
                    best_match = (pg2, t2)
                    best_diffs = diffs
                    best_diff_html = diff_html
            
            # If we found a good match
            if best_match and best_sim >= self.diff_threshold:
                pg2, t2 = best_match
                status = "moved" if pg1 != pg2 else "modified" if best_diffs > 0 else "matched"
                
                results.append(self._build_table_diff(
                    status, pg1, pg2, t1, t2,
                    differences=best_diffs,
                    similarity=best_sim,
                    diff_html=best_diff_html if best_diffs > 0 else ""
                ))
                matched_tables2.add((pg2, t2["table_id"]))
            else:
                # No good match found, table only in first PDF
                results.append(self._build_table_diff("deleted", pg1, None, t1, None))
        
        # Add remaining unmatched tables from second PDF
        for pg2, t2 in remaining2:
            if (pg2, t2["table_id"]) not in matched_tables2:
                results.append(self._build_table_diff("inserted", None, pg2, None, t2))
        
        logger.info(f"Matched tables: {len(results) - (len(tables1) + len(tables2) - len(matched_tables2) * 2)}")
        return results
    
    def _match_by_content_hash(self, tables1, tables2) -> List[Tuple[Tuple, Tuple]]:
        """Match tables using content hash for exact content matching."""
        matches = []
        
        # Build lookup by content hash for second PDF's tables
        hash_to_table2 = defaultdict(list)
        for pg, table in tables2:
            if "content_hash" in table and table["content_hash"]:
                hash_to_table2[table["content_hash"]].append((pg, table))
        
        # Find exact matches
        for pg1, t1 in tables1:
            if "content_hash" in t1 and t1["content_hash"]:
                for pg2, t2 in hash_to_table2.get(t1["content_hash"], []):
                    matches.append(((pg1, t1), (pg2, t2)))
                    break  # Only match each table once
        
        return matches
    
    def _compare_tables(self, table1: List[List[str]], table2: List[List[str]]) -> Tuple[float, int, str]:
        """
        Compare two tables and return similarity score, difference count, and HTML diff.
        
        Returns:
            Tuple of (similarity_score, differences_count, diff_html)
        """
        if not table1 or not table2:
            return 0.0, 0, ""
        
        # Calculate overall similarity
        str_a = "\n".join("∥".join(r) for r in table1)
        str_b = "\n".join("∥".join(r) for r in table2)
        similarity = SequenceMatcher(None, str_a, str_b).ratio()
        
        # If tables are very different, just mark them as such
        if similarity < self.diff_threshold:
            return similarity, max(len(table1), len(table2)), self._generate_diff_html(table1, table2)
        
        # Cell-by-cell comparison for detailed differences
        differences = 0
        diff_html = ""
        
        # Generate an HTML diff with cell-level highlighting
        if similarity < 1.0:
            diff_html = self._generate_diff_html(table1, table2)
            
            # Count differences (cells that don't match)
            differences = self._count_cell_differences(table1, table2)
        
        return similarity, differences, diff_html
    
    def _count_cell_differences(self, table1: List[List[str]], table2: List[List[str]]) -> int:
        """Count the number of differing cells between two tables."""
        diff_count = 0
        rows = max(len(table1), len(table2))
        
        for i in range(rows):
            # Get rows or empty lists if row index exceeds table size
            row1 = table1[i] if i < len(table1) else []
            row2 = table2[i] if i < len(table2) else []
            
            # Compare cells in this row
            cols = max(len(row1), len(row2))
            for j in range(cols):
                cell1 = row1[j] if j < len(row1) else ""
                cell2 = row2[j] if j < len(row2) else ""
                
                # Normalize cell content for comparison
                norm_cell1 = self._normalize_cell(cell1)
                norm_cell2 = self._normalize_cell(cell2)
                
                # Check if cells differ
                if norm_cell1 != norm_cell2:
                    # Check for fuzzy match
                    if not self._fuzzy_match_cells(norm_cell1, norm_cell2):
                        diff_count += 1
        
        return diff_count
    
    def _normalize_cell(self, cell: str) -> str:
        """Normalize cell content for better comparison."""
        if not cell:
            return ""
        # Remove extra whitespace and convert to lowercase
        return re.sub(r'\s+', ' ', cell.strip().lower())
    
    def _fuzzy_match_cells(self, cell1: str, cell2: str) -> bool:
        """Check if two cells are similar enough to be considered matching."""
        if not cell1 and not cell2:
            return True
        if not cell1 or not cell2:
            return False
            
        # Check if one is a subset of the other
        if cell1 in cell2 or cell2 in cell1:
            return True
            
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, cell1, cell2).ratio() >= self.cell_match_threshold
    
    def _build_table_diff(self, status, pg1, pg2, t1, t2, differences=0, similarity=0.0, diff_html=""):
        """Create a table difference record."""
        # For matched tables, we don't need to include the diff HTML
        if status in ("matched", "moved") and not differences:
            diff_html = ""
        elif status == "deleted":
            diff_html = self._format_table_as_deleted(t1["content"])
        elif status == "inserted":
            diff_html = self._format_table_as_inserted(t2["content"])
        # For modified, the diff_html should already be set
            
        return {
            "status": status,
            "page1": pg1,
            "page2": pg2,
            "table_id1": t1.get("table_id") if t1 else None,
            "table_id2": t2.get("table_id") if t2 else None,
            "differences": differences,
            "similarity": similarity,
            "diff_html": diff_html,
            "has_nested_tables": t1.get("has_nested_tables", False) if t1 else (t2.get("has_nested_tables", False) if t2 else False),
            "nested_tables": t1.get("nested_tables", []) if t1 else (t2.get("nested_tables", []) if t2 else [])
        }
    
    # ─── text comparison ──────────────────────────────────────────
    def _compare_text_elements(self, elems1: List[Dict], elems2: List[Dict]):
        """Compare text elements between PDFs with improved accuracy."""
        lines1 = [e["content"] for e in elems1]
        lines2 = [e["content"] for e in elems2]
        
        diff_results = []
        
        # Use SequenceMatcher for line-by-line comparison
        sm = SequenceMatcher(None, lines1, lines2)
        
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                # Lines match exactly
                for k in range(i1, i2):
                    diff_results.append(self._make_text_diff(
                        "equal", k, j1 + k - i1,
                        lines1[k], lines2[j1 + k - i1]
                    ))
            elif tag == "replace":
                # Lines were changed - try to highlight specific differences
                replaced_lines = max(i2 - i1, j2 - j1)
                for k in range(replaced_lines):
                    l1 = lines1[i1 + k] if i1 + k < i2 else ""
                    l2 = lines2[j1 + k] if j1 + k < j2 else ""
                    
                    # Check for fuzzy matches and generate inline diff
                    if l1 and l2 and SequenceMatcher(None, l1, l2).ratio() >= self.fuzzy_match_threshold:
                        diff_html = self._generate_inline_text_diff(l1, l2)
                        status = "modified"
                    else:
                        diff_html = ""
                        status = "changed"
                        
                    diff_results.append(self._make_text_diff(
                        status,
                        i1 + k if l1 else None,
                        j1 + k if l2 else None,
                        l1, l2, diff_html
                    ))
            elif tag == "delete":
                # Lines only in first document
                for k in range(i1, i2):
                    diff_results.append(self._make_text_diff(
                        "deleted", k, None, lines1[k], ""
                    ))
            elif tag == "insert":
                # Lines only in second document
                for k in range(j1, j2):
                    diff_results.append(self._make_text_diff(
                        "inserted", None, k, "", lines2[k]
                    ))
                    
        return diff_results
    
    def _make_text_diff(self, status, n1, n2, t1, t2, diff_html=""):
        """Create a text difference record."""
        return {
            "status": status, 
            "line_num1": n1, 
            "line_num2": n2,
            "text1": t1, 
            "text2": t2, 
            "diff_html": diff_html
        }
    
    def _generate_inline_text_diff(self, text1: str, text2: str) -> str:
        """Generate HTML with inline highlighting of text differences."""
        # Split into words for more accurate diff
        words1 = text1.split()
        words2 = text2.split()
        
        # Use SequenceMatcher for word-by-word comparison
        sm = SequenceMatcher(None, words1, words2)
        
        result1 = []
        result2 = []
        
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                # Words match
                segment = ' '.join(words1[i1:i2])
                result1.append(segment)
                result2.append(segment)
            elif tag == 'replace':
                # Words changed
                result1.append(f'<span class="diff-deleted">{" ".join(words1[i1:i2])}</span>')
                result2.append(f'<span class="diff-inserted">{" ".join(words2[j1:j2])}</span>')
            elif tag == 'delete':
                # Words only in first text
                result1.append(f'<span class="diff-deleted">{" ".join(words1[i1:i2])}</span>')
            elif tag == 'insert':
                # Words only in second text
                result2.append(f'<span class="diff-inserted">{" ".join(words2[j1:j2])}</span>')
        
        # Combine results into two-column format
        return f'<div class="inline-diff"><div class="diff-old">{" ".join(result1)}</div>' + \
               f'<div class="diff-new">{" ".join(result2)}</div></div>'
    
    # ─── HTML diff generation ─────────────────────────────────────
    def _generate_diff_html(self, table1: List[List[str]], table2: List[List[str]]) -> str:
        """Generate HTML showing differences between two tables with column-level highlighting."""
        max_rows = max(len(table1), len(table2))
        
        # Determine the maximum number of columns
        max_cols = 0
        for row in table1 + table2:
            max_cols = max(max_cols, len(row))
        
        # Create side-by-side table diff
        html = ['<div class="table-diff-container">']
        
        # Table 1 (left side)
        html.append('<div class="table-diff-left"><table class="diff-table">')
        for i in range(max_rows):
            html.append('<tr>')
            row = table1[i] if i < len(table1) else []
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_class = ""
                
                # Determine if this cell exists in the second table
                has_match = False
                if i < len(table2) and j < len(table2[i]):
                    other_cell = table2[i][j]
                    # Check if cells match
                    if self._normalize_cell(cell) == self._normalize_cell(other_cell) or \
                       self._fuzzy_match_cells(self._normalize_cell(cell), self._normalize_cell(other_cell)):
                        has_match = True
                
                if not has_match and cell.strip():
                    cell_class = ' class="diff-deleted"'
                
                html.append(f'<td{cell_class}>{self._escape_html(cell)}</td>')
            html.append('</tr>')
        html.append('</table></div>')
        
        # Table 2 (right side)
        html.append('<div class="table-diff-right"><table class="diff-table">')
        for i in range(max_rows):
            html.append('<tr>')
            row = table2[i] if i < len(table2) else []
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_class = ""
                
                # Determine if this cell exists in the first table
                has_match = False
                if i < len(table1) and j < len(table1[i]):
                    other_cell = table1[i][j]
                    # Check if cells match
                    if self._normalize_cell(cell) == self._normalize_cell(other_cell) or \
                       self._fuzzy_match_cells(self._normalize_cell(cell), self._normalize_cell(other_cell)):
                        has_match = True
                
                if not has_match and cell.strip():
                    cell_class = ' class="diff-inserted"'
                
                html.append(f'<td{cell_class}>{self._escape_html(cell)}</td>')
            html.append('</tr>')
        html.append('</table></div>')
        
        html.append('</div>')
        return "".join(html)
    
    def _format_table_as_deleted(self, table):
        """Format a table that only exists in the first document."""
        html = ['<table class="diff-table deleted">']
        for row in table:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td class="deleted">{self._escape_html(cell)}</td>')
            html.append('</tr>')
        html.append('</table>')
        return "".join(html)

    def _format_table_as_inserted(self, table):
        """Format a table that only exists in the second document."""
        html = ['<table class="diff-table inserted">']
        for row in table:
            html.append('<tr>')
            for cell in row:
                html.append(f'<td class="inserted">{self._escape_html(cell)}</td>')
            html.append('</tr>')
        html.append('</table>')
        return "".join(html)
    
    @staticmethod
    def _escape_html(text):
        """Escape HTML special characters in text."""
        if not text:
            return ""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")