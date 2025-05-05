
import logging
import re
import itertools
import hashlib
import functools
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher, get_close_matches
from typing import Dict, List, Tuple, Any, Optional, Set, Union
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PdfCompare:
    def __init__(self, 
                 diff_threshold: float = 0.75,
                 cell_match_threshold: float = 0.9,
                 fuzzy_match_threshold: float = 0.8,
                 semantic_similarity_threshold: float = 0.85,
                 max_workers: int = 4):
        self.diff_threshold = diff_threshold        # Table similarity threshold
        self.cell_match_threshold = cell_match_threshold  # Cell content similarity threshold  
        self.fuzzy_match_threshold = fuzzy_match_threshold  # Fuzzy text matching threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold  # Semantic matching threshold
        
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Configure parallelization
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize caches
        self.embedding_cache = {}
        self.table_comparison_cache = {}

    # ─── public ───────────────────────────────────────────────────
    def compare_pdfs(self, pdf1: Dict, pdf2: Dict) -> Dict:
        """
        Compare two PDFs and generate detailed differences using content-based matching.
        
        Args:
            pdf1: First PDF structure from extractor
            pdf2: Second PDF structure from extractor
            
        Returns:
            Dictionary with comparison results
        """
        logger.info("Starting PDF comparison with content-based matching")
        max_pages = max(len(pdf1), len(pdf2))
        results = {"max_pages": max_pages, "pages": {}}

        # Pre-process: collect all tables and text from both PDFs with page information
        tables1 = list(self._collect_tables(pdf1))
        tables2 = list(self._collect_tables(pdf2))
        
        text_blocks1 = self._collect_text_blocks(pdf1)
        text_blocks2 = self._collect_text_blocks(pdf2)
        
        logger.info(f"Found {len(tables1)} tables in first PDF and {len(tables2)} tables in second PDF")
        
        # Match tables across both PDFs using global matching (ignoring page position)
        table_matches = self._match_tables_global(tables1, tables2)
        
        # Match text blocks across all pages
        text_matches = self._match_text_blocks_global(text_blocks1, text_blocks2)
        
        # Organize matches by page for the report
        tables_by_page = self._organize_table_matches_by_page(table_matches)
        text_by_page = self._organize_text_matches_by_page(text_matches)
        
        # Process each page
        for p in range(1, max_pages + 1):
            # Get table differences for this page
            table_diff = tables_by_page.get(p, [])
            
            # Get text differences for this page
            text_diff = text_by_page.get(p, [])
            
            # Store results for this page
            results["pages"][p] = {
                "text_differences": text_diff,
                "table_differences": table_diff
            }

        logger.info("PDF comparison completed")
        return results

    # ─── text processing ──────────────────────────────────────────
    def _collect_text_blocks(self, pdf_data):
        """Extract all text elements from PDF data with page information."""
        text_blocks = []
        for pg, pdata in pdf_data.items():
            for idx, elem in enumerate(pdata.get("elements", [])):
                if elem.get("type") == "text":
                    block = {
                        "page": pg,
                        "content": elem["content"],
                        "index": idx,
                        "element": elem,
                        "hash": hashlib.md5(elem["content"].encode()).hexdigest()
                    }
                    text_blocks.append(block)
        return text_blocks
    
    def _match_text_blocks_global(self, blocks1, blocks2):
        """Match text blocks across all pages using content similarity."""
        matches = []
        matched_blocks2 = set()
        
        for block1 in blocks1:
            best_match = None
            best_score = self.fuzzy_match_threshold
            
            for i, block2 in enumerate(blocks2):
                if i in matched_blocks2:
                    continue
                    
                # Check for exact match first
                if block1["hash"] == block2["hash"]:
                    score = 1.0
                else:
                    # Calculate similarity with a more accurate algorithm
                    # This is a key fix for proper text difference detection
                    content1 = block1["content"].strip()
                    content2 = block2["content"].strip()
                    
                    # Use SequenceMatcher for more accurate difference detection
                    score = SequenceMatcher(None, content1, content2).ratio()
                    
                    # Improve scoring for partially matching text
                    # This helps identify text that has been edited slightly
                    if score < self.fuzzy_match_threshold:
                        # Check if one is a subset of the other (insertions or deletions)
                        if content1 in content2 or content2 in content1:
                            # Boost score for partial matches
                            score = max(score, self.fuzzy_match_threshold + 0.05)
                
                if score > best_score:
                    best_score = score
                    best_match = (i, block2)
            
            # Determine status with enhanced logic
            if best_match:
                i, block2 = best_match
                
                # Better classification of changes
                if block1["page"] != block2["page"]:
                    status = "moved"  # Text moved to a different page
                elif best_score == 1.0:
                    status = "matched"  # Exact match
                elif best_score >= 0.95:
                    status = "similar"  # Very similar, minor changes
                else:
                    status = "modified"  # Significant changes
                    
                matches.append({
                    "block1": block1,
                    "block2": block2,
                    "status": status,
                    "score": best_score,
                    "page1": block1["page"],
                    "page2": block2["page"]
                })
                matched_blocks2.add(i)
            else:
                # No match found - text only in first PDF
                matches.append({
                    "block1": block1,
                    "block2": None,
                    "status": "deleted",
                    "score": 0.0,
                    "page1": block1["page"],
                    "page2": None
                })
        
        # Add text blocks that only exist in second PDF
        for i, block2 in enumerate(blocks2):
            if i not in matched_blocks2:
                matches.append({
                    "block1": None,
                    "block2": block2,
                    "status": "inserted",
                    "score": 0.0,
                    "page1": None,
                    "page2": block2["page"]
                })
        
        return matches
    
    def _organize_text_matches_by_page(self, text_matches):
        """Organize text matches by page for report generation."""
        text_by_page = defaultdict(list)
        
        for match in text_matches:
            # Add to source page
            if match["page1"]:
                text_by_page[match["page1"]].append({
                    "status": match["status"],
                    "text1": match["block1"]["content"] if match["block1"] else "",
                    "text2": match["block2"]["content"] if match["block2"] else "",
                    "score": match["score"],
                    "page1": match["page1"],
                    "page2": match["page2"]
                })
            
            # Add to target page if moved
            if match["status"] in ["moved", "inserted"] and match["page2"] and match["page2"] != match.get("page1"):
                text_by_page[match["page2"]].append({
                    "status": match["status"],
                    "text1": match["block1"]["content"] if match["block1"] else "",
                    "text2": match["block2"]["content"] if match["block2"] else "",
                    "score": match["score"],
                    "page1": match["page1"],
                    "page2": match["page2"]
                })
        
        return text_by_page

    # ─── table processing ─────────────────────────────────────────
    def _collect_tables(self, pdf_data):
        """Extract all tables from PDF data with page information."""
        for pg, pdata in pdf_data.items():
            for el in pdata.get("elements", []):
                if el.get("type") == "table":
                    yield (pg, el)

    def _match_tables_global(self, tables1, tables2) -> List[Dict]:
        """
        Match tables between two PDFs using content-based matching, ignoring page positions.
        
        This enhanced algorithm:
        1. Uses content hashes for exact matches
        2. Uses semantic similarity for similar content
        3. Properly handles multi-page tables
        4. Avoids duplicate reporting
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
        
        # Step 1: Match by exact content hash (exact content match)
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
        
        # Step 2: For remaining tables, use similarity matching
        remaining1 = [(pg, t) for pg, t in tables1 
                     if not any((pg, t) == match[0] for match in content_matches)]
        remaining2 = [(pg, t) for pg, t in tables2 
                     if (pg, t["table_id"]) not in matched_tables2]
        
        # Step 3: Compute semantic embeddings for all remaining tables
        semantic_matches = self._match_tables_by_similarity(remaining1, remaining2)
        
        # Process semantic matches
        for (pg1, t1), (pg2, t2), similarity in semantic_matches:
            status = "moved" if pg1 != pg2 else "modified"
            
            # Get detailed differences
            _, diffs, diff_html = self._compare_tables(t1["content"], t2["content"])
            
            results.append(self._build_table_diff(
                status, pg1, pg2, t1, t2,
                differences=diffs,
                similarity=similarity,
                diff_html=diff_html if diffs > 0 else ""
            ))
            matched_tables2.add((pg2, t2["table_id"]))
        
        # Add remaining unmatched tables
        for pg1, t1 in remaining1:
            if not any((pg1, t1) == match[0] for match in content_matches) and \
               not any((pg1, t1) == match[0] for match in semantic_matches):
                results.append(self._build_table_diff("deleted", pg1, None, t1, None))
                
        for pg2, t2 in remaining2:
            if (pg2, t2["table_id"]) not in matched_tables2:
                results.append(self._build_table_diff("inserted", None, pg2, None, t2))
        
        logger.info(f"Matched tables: {len(content_matches) + len(semantic_matches)}")
        return results
    
    def _organize_table_matches_by_page(self, table_matches):
        """
        Organize table matches by page for report generation.
        Avoids duplicate table entries in the report.
        """
        tables_by_page = defaultdict(list)
        
        # Keep track of table IDs already added to avoid duplicates
        added_tables = set()
        
        for diff in table_matches:
            # Create a unique identifier for this table comparison
            table_key = (diff.get("table_id1"), diff.get("table_id2"), diff.get("status"))
            
            # Skip if already added
            if table_key in added_tables:
                continue
                
            added_tables.add(table_key)
            
            # Add to source page
            if diff.get("page1") is not None:
                tables_by_page[diff.get("page1")].append(diff)
            
            # Add to target page only if it's a move to a different page
            if diff.get("status") == "moved" and diff.get("page2") is not None and diff.get("page1") != diff.get("page2"):
                # Create a copy for the target page with appropriate status
                moved_copy = diff.copy()
                moved_copy["status"] = "moved_to"
                tables_by_page[diff.get("page2")].append(moved_copy)
        
        return tables_by_page
    
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
    
    def _match_tables_by_similarity(self, tables1, tables2):
        """
        Match tables using multiple similarity metrics regardless of page position.
        Returns list of (table1, table2, similarity_score) tuples.
        """
        if not tables1 or not tables2:
            return []
            
        matches = []
        
        # Create a similarity matrix for all possible pairs
        similarity_matrix = []
        all_similarities = []
        
        # For each table in first PDF
        for i, (pg1, t1) in enumerate(tables1):
            similarities = []
            
            # Compare with each table in second PDF
            for j, (pg2, t2) in enumerate(tables2):
                # Calculate similarity using multiple metrics
                similarity = self._calculate_table_similarity(t1, t2)
                
                # Store similarity with table indices
                similarities.append((similarity, j))
                all_similarities.append((similarity, i, j))
                
            # Sort by similarity (highest first)
            similarities.sort(reverse=True)
            similarity_matrix.append(similarities)
        
        # Sort all similarities globally to ensure we match the most similar tables first
        all_similarities.sort(reverse=True)
        
        # Use a global matching algorithm to find optimal matches
        used_tables1 = set()
        used_tables2 = set()
        
        # Process matches in order of decreasing similarity
        for sim, i, j in all_similarities:
            # Only consider sufficiently similar tables
            if sim < self.semantic_similarity_threshold:
                continue
                
            # Skip if either table is already matched
            if i in used_tables1 or j in used_tables2:
                continue
                
            # Add this match
            pg1, t1 = tables1[i]
            pg2, t2 = tables2[j]
            
            matches.append(((pg1, t1), (pg2, t2), sim))
            
            # Mark both tables as used
            used_tables1.add(i)
            used_tables2.add(j)
        
        return matches
    
    def _calculate_table_similarity(self, t1, t2):
        """
        Calculate overall similarity between two tables using multiple metrics:
        1. Content similarity
        2. Structure similarity
        3. Semantic similarity
        """
        # 1. Check if tables have content
        if not t1.get("content") or not t2.get("content"):
            return 0.0
            
        content1 = t1["content"]
        content2 = t2["content"]
        
        if not content1 or not content2:
            return 0.0
        
        # 2. Check if tables are identical using content hash
        if t1.get("content_hash") and t2.get("content_hash") and t1["content_hash"] == t2["content_hash"]:
            return 1.0
            
        # 3. Structure similarity - check if tables have similar dimensions
        row_count1 = len(content1)
        row_count2 = len(content2)
        row_ratio = min(row_count1, row_count2) / max(row_count1, row_count2) if max(row_count1, row_count2) > 0 else 0
        
        # Compare column count
        col_count1 = len(content1[0]) if content1 and content1[0] else 0
        col_count2 = len(content2[0]) if content2 and content2[0] else 0
        col_ratio = min(col_count1, col_count2) / max(col_count1, col_count2) if max(col_count1, col_count2) > 0 else 0
        
        # 4. Cell content similarity
        cell_similarity = self._calculate_cell_content_similarity(content1, content2)
        
        # 5. Semantic similarity - compare table meaning
        semantic_similarity = 0.5  # Default value
        table1_text = self._table_to_text(t1)
        table2_text = self._table_to_text(t2)
        
        if table1_text and table2_text:
            try:
                # Generate embeddings
                emb1 = self._get_table_embedding(table1_text)
                emb2 = self._get_table_embedding(table2_text)
                
                # Calculate cosine similarity
                semantic_similarity = np.dot(emb1, emb2)
                
                # Normalize to 0-1 range
                semantic_similarity = max(0.0, min(1.0, semantic_similarity))
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {str(e)}")
        
        # Combine similarities with weights
        combined_similarity = (
            cell_similarity * 0.6 +       # Cell content is most important
            row_ratio * 0.1 +             # Row structure matters somewhat
            col_ratio * 0.1 +             # Column structure matters somewhat
            semantic_similarity * 0.2     # Overall meaning matters
        )
        
        return combined_similarity
    
    def _calculate_cell_content_similarity(self, content1, content2):
        """Calculate similarity between table cells."""
        # Limit for performance
        max_rows = min(len(content1), len(content2), 20)
        max_cols = min(
            len(content1[0]) if content1 and content1[0] else 0,
            len(content2[0]) if content2 and content2[0] else 0,
            10
        )
        
        if max_rows == 0 or max_cols == 0:
            return 0.0
        
        # Compare cells
        matching_cells = 0
        total_cells = max_rows * max_cols
        
        for i in range(max_rows):
            for j in range(max_cols):
                try:
                    cell1 = content1[i][j] if i < len(content1) and j < len(content1[i]) else ""
                    cell2 = content2[i][j] if i < len(content2) and j < len(content2[i]) else ""
                    
                    # Skip empty cells
                    if not cell1.strip() and not cell2.strip():
                        total_cells -= 1
                        continue
                    
                    # Check for exact match
                    if self._normalize_cell(cell1) == self._normalize_cell(cell2):
                        matching_cells += 1
                    # Check for fuzzy match
                    elif self._fuzzy_match_cells(self._normalize_cell(cell1), self._normalize_cell(cell2)):
                        matching_cells += 0.8  # Partial credit
                except IndexError:
                    total_cells -= 1
        
        return matching_cells / max(1, total_cells)

    def _normalize_cell(self, cell: str) -> str:
        """Normalize cell content for better comparison."""
        if not cell:
            return ""
        # Remove extra whitespace and convert to lowercase
        return re.sub(r'\s+', ' ', str(cell).strip().lower())

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
        similarity = SequenceMatcher(None, cell1, cell2).ratio()
        
        # Check for numeric similarity
        if similarity < self.cell_match_threshold:
            # Extract numbers from cells
            num1 = self._extract_numeric_value(cell1)
            num2 = self._extract_numeric_value(cell2)
            
            if num1 is not None and num2 is not None:
                # If the numbers are close, consider it a match
                if max(num1, num2) > 0:
                    relative_diff = abs(num1 - num2) / max(num1, num2)
                    if relative_diff < 0.05:  # 5% tolerance
                        return True
        
        return similarity >= self.cell_match_threshold

    def _extract_numeric_value(self, text):
        """Extract numeric value from text."""
        if not text:
            return None
            
        # Try to find and parse a number in the text
        matches = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return None

    def _table_to_text(self, table):
        """Convert table to text representation for semantic matching."""
        if not table.get("content"):
            return ""
        
        # Join cells with spaces, rows with newlines
        return "\n".join(" ".join(str(cell) for cell in row if cell) 
                        for row in table["content"] if any(row))

    @functools.lru_cache(maxsize=128)
    def _get_table_embedding(self, table_content_str):
        """Get and cache table embeddings."""
        # If already in cache, return it
        hash_key = hashlib.md5(table_content_str.encode()).hexdigest()
        if hash_key in self.embedding_cache:
            return self.embedding_cache[hash_key]
            
        # Otherwise, compute and cache
        embedding = self.embedding_model.encode([table_content_str])[0]
        self.embedding_cache[hash_key] = embedding
        return embedding
    
    def _compare_tables(self, table1: List[List[str]], table2: List[List[str]]) -> Tuple[float, int, str]:
        """
        Compare two tables and return similarity score, difference count, and HTML diff.
        
        Returns:
            Tuple of (similarity_score, differences_count, diff_html)
        """
        if not table1 or not table2:
            return 0.0, 0, ""
        
        # Create a cache key for this comparison
        cache_key = (
            hashlib.md5(str(table1).encode()).hexdigest(),
            hashlib.md5(str(table2).encode()).hexdigest()
        )
        
        # Check if we have this comparison in cache
        if cache_key in self.table_comparison_cache:
            return self.table_comparison_cache[cache_key]
        
        # Calculate overall similarity
        str_a = "\n".join("∥".join(str(r) for r in row) for row in table1)
        str_b = "\n".join("∥".join(str(r) for r in row) for row in table2)
        similarity = SequenceMatcher(None, str_a, str_b).ratio()
        
        # If tables are very different, just mark them as such
        if similarity < self.diff_threshold:
            result = (similarity, max(len(table1), len(table2)), self._generate_diff_html(table1, table2))
            self.table_comparison_cache[cache_key] = result
            return result
        
        # Cell-by-cell comparison for detailed differences
        differences = 0
        diff_html = ""
        
        # Generate an HTML diff with cell-level highlighting
        if similarity < 1.0:
            diff_html = self._generate_diff_html(table1, table2)
            
            # Count differences (cells that don't match)
            differences = self._count_cell_differences(table1, table2)
        
        result = (similarity, differences, diff_html)
        self.table_comparison_cache[cache_key] = result
        return result

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

    def _generate_diff_html(self, table1: List[List[str]], table2: List[List[str]]) -> str:
        """Generate HTML showing differences between two tables."""
        # Use side-by-side diff
        return self._generate_side_by_side_diff(table1, table2)

    def _generate_side_by_side_diff(self, table1: List[List[str]], table2: List[List[str]]) -> str:
        """Generate side-by-side HTML diff of two tables."""
        max_rows = max(len(table1), len(table2))
        max_cols = max(
            max((len(row) for row in table1), default=0),
            max((len(row) for row in table2), default=0)
        )
        
        html = ['<div class="table-diff-container">']
        
        # Table 1 (Left side)
        html.append('<div class="table-diff-left">')
        html.append(f'<table class="diff-table source1" cellspacing="0" cellpadding="3">')
        html.append(self._generate_table_html(table1, table2, max_rows, max_cols, is_left=True))
        html.append('</table>')
        html.append('</div>')
        
        # Table 2 (Right side)
        html.append('<div class="table-diff-right">')
        html.append(f'<table class="diff-table source2" cellspacing="0" cellpadding="3">')
        html.append(self._generate_table_html(table2, table1, max_rows, max_cols, is_left=False))
        html.append('</table>')
        html.append('</div>')
        
        html.append('</div>')
        
        return ''.join(html)

    def _generate_table_html(self, source_table, compare_table, max_rows, max_cols, is_left):
        """Generate HTML for a single table with proper cell highlighting."""
        html = []
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i in range(max_rows):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')
            
            # Get row or empty list
            row = source_table[i] if i < len(source_table) else []
            compare_row = compare_table[i] if i < len(compare_table) else []
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                compare_cell = compare_row[j] if j < len(compare_row) else ""
                
                cell_html = self._escape_html(str(cell))
                cell_class = "empty" if not str(cell).strip() else ""
                
                # Determine cell status
                if not str(cell).strip() and not str(compare_cell).strip():
                    cell_class = ""  # Both empty
                elif not str(cell).strip():
                    cell_class = "empty"  # This cell is empty
                elif not str(compare_cell).strip():
                    cell_class = "inserted" if not is_left else "deleted"
                elif self._normalize_cell(cell) == self._normalize_cell(compare_cell):
                    cell_class = "similar"
                elif self._fuzzy_match_cells(self._normalize_cell(cell), self._normalize_cell(compare_cell)):
                    cell_class = "similar"
                else:
                    cell_class = "modified"
                
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        return ''.join(html)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters in text."""
        if not text:
            return ""
            
        return (str(text).replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;"))
        
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
        
    def _format_table_as_deleted(self, table_content: List[List[str]]) -> str:
        """Format a table that only exists in the first document."""
        if not table_content:
            return ""
            
        html = ['<div class="table-diff-container">']
        html.append('<div class="table-diff-left" style="width:100%">') 
        html.append('<table class="diff-table deleted" cellspacing="0" cellpadding="3">')
        
        max_cols = max((len(row) for row in table_content), default=0)
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(str(cell))
                cell_class = "deleted"  # All cells are deleted
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        html.append('</div>')
        
        return ''.join(html)

    def _format_table_as_inserted(self, table_content: List[List[str]]) -> str:
        """Format a table that only exists in the second document."""
        if not table_content:
            return ""
            
        html = ['<div class="table-diff-container">']
        html.append('<div class="table-diff-right" style="width:100%">')
        html.append('<table class="diff-table inserted" cellspacing="0" cellpadding="3">')
        
        max_cols = max((len(row) for row in table_content), default=0)
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(str(cell))
                cell_class = "inserted"  # All cells are inserted
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        html.append('</div>')
        
        return ''.join(html)    
    
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
