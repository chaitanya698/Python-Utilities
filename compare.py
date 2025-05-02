"""
compare.py
----------
Enhanced diff engine for text & table comparison.

Key improvements:
* Semantic similarity-based matching for tables regardless of page position
* Support for multi-page table comparison
* Improved diff visualization with column-level highlighting
* Detailed cell-level difference detection
* Fuzzy and semantic matching for higher accuracy
* Improved handling of tables with missing borders
"""

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
        
        # Match tables across both PDFs using global matching (ignoring page position)
        table_matches = self._match_tables_global(tables1, tables2)
        
        # Organize table matches by page for the report
        tables_by_page = defaultdict(list)
        for diff in table_matches:
            # For regular tables
            if diff.get("page1") is not None:
                tables_by_page[diff.get("page1")].append(diff)
            if diff.get("page2") is not None and diff.get("page1") != diff.get("page2"):
                tables_by_page[diff.get("page2")].append(diff)

        # Process each page
        for p in range(1, max_pages + 1):
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
        Match tables between two PDFs using content-based and semantic similarity.
        
        This enhanced algorithm:
        1. Prioritizes content matching over position matching
        2. Uses semantic similarity for matching tables with similar content
        3. Handles multi-page tables correctly
        4. Provides detailed cell-level diff information
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
        
        # Step 2: For remaining tables, use fuzzy similarity matching
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
    
    def _match_tables_by_similarity(self, tables1, tables2):
        """
        Match tables using multiple similarity metrics regardless of page position.
        Returns list of (table1, table2, similarity_score) tuples.
        """
        if not tables1 or not tables2:
            return []
            
        matches = []
        similarity_matrix = []
        
        # For each table in first PDF
        for pg1, t1 in tables1:
            similarities = []
            
            # Compare with each table in second PDF
            for pg2, t2 in tables2:
                # Calculate similarity using multiple metrics
                similarity = self._calculate_table_similarity(t1, t2)
                similarities.append((similarity, pg2, t2))
                
            # Sort by similarity (highest first)
            similarities.sort(reverse=True)
            similarity_matrix.append(similarities)
        
        # Use a greedy algorithm to select best non-conflicting matches
        used_tables2 = set()
        
        # Process in order of similarity
        for i, sims in enumerate(similarity_matrix):
            for sim, pg2, t2 in sims:
                if sim >= self.semantic_similarity_threshold and (pg2, t2["table_id"]) not in used_tables2:
                    matches.append((tables1[i], (pg2, t2), sim))
                    used_tables2.add((pg2, t2["table_id"]))
                    break
        
        return matches
    
    def _calculate_table_similarity(self, t1, t2):
        """
        Calculate overall similarity between two tables using multiple metrics:
        1. Content similarity
        2. Structure similarity
        3. Semantic similarity
        """
        # 1. Content similarity - check if cells have similar text
        if not t1.get("content") or not t2.get("content"):
            return 0.0
            
        # 2. Structure similarity - check if tables have similar dimensions
        content1 = t1["content"]
        content2 = t2["content"]
        
        if not content1 or not content2:
            return 0.0
            
        # Compare row count
        row_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
        
        # Compare column count
        col_count1 = len(content1[0]) if content1 and content1[0] else 0
        col_count2 = len(content2[0]) if content2 and content2[0] else 0
        col_ratio = min(col_count1, col_count2) / max(col_count1, col_count2) if max(col_count1, col_count2) > 0 else 0
        
        # 3. Cell content similarity
        cell_similarity = self._calculate_cell_content_similarity(content1, content2)
        
        # 4. Semantic similarity - compare table embeddings if available
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
            except Exception as e:
                logger.warning(f"Error calculating semantic similarity: {str(e)}")
        
        # Combine similarities with weights - prioritizing content over position
        combined_similarity = (
            cell_similarity * 0.7 +  # Increased weight for content similarity
            row_ratio * 0.05 +      # Decreased weight for row structure
            col_ratio * 0.05 +      # Decreased weight for column structure
            semantic_similarity * 0.2  # Maintain weight for semantic similarity
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
    
    def _table_to_text(self, table):
        """Convert table to text representation for semantic matching."""
        if not table.get("content"):
            return ""
        
        # Join cells with spaces, rows with newlines
        return "\n".join(" ".join(cell for cell in row if cell) 
                         for row in table["content"] if any(row))
    
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
        str_a = "\n".join("∥".join(r) for r in table1)
        str_b = "\n".join("∥".join(r) for r in table2)
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
        """Compare text elements between PDFs with enhanced accuracy."""
        # Extract lines of text
        lines1 = [e["content"] for e in elems1]
        lines2 = [e["content"] for e in elems2]
        
        # Perform semantic alignment for text blocks to improve matching
        if lines1 and lines2:
            try:
                # Group text into logical blocks for more accurate semantic comparison
                blocks1 = self._group_text_into_blocks(lines1)
                blocks2 = self._group_text_into_blocks(lines2)
                
                # Generate embeddings for blocks
                block_embeddings1 = self.embedding_model.encode(blocks1)
                block_embeddings2 = self.embedding_model.encode(blocks2)
                
                # Find block matches
                block_matches = self._match_text_blocks(blocks1, blocks2, block_embeddings1, block_embeddings2)
                
                # Apply block matches to improve line matching
                aligned_lines1, aligned_lines2 = self._align_text_based_on_blocks(lines1, lines2, blocks1, blocks2, block_matches)
                
                # Use aligned lines if successful
                if aligned_lines1 and aligned_lines2:
                    lines1, lines2 = aligned_lines1, aligned_lines2
            except Exception as e:
                logger.warning(f"Error in semantic text matching: {str(e)}")
        
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
                    
                    # Check for fuzzy matches and semantic similarity
                    if l1 and l2:
                        seq_similarity = SequenceMatcher(None, l1, l2).ratio()
                        
                        if seq_similarity >= self.fuzzy_match_threshold:
                            # Close enough for inline diff
                            diff_html = self._generate_inline_text_diff(l1, l2)
                            status = "modified"
                        else:
                            # Check for semantic similarity
                            try:
                                emb1 = self.embedding_model.encode([l1])[0]
                                emb2 = self.embedding_model.encode([l2])[0]
                                sem_similarity = np.dot(emb1, emb2)
                                
                                if sem_similarity >= self.semantic_similarity_threshold:
                                    diff_html = f'<div class="inline-diff semantic"><div class="diff-old">{l1}</div><div class="diff-new">{l2}</div></div>'
                                    status = "similar"
                                else:
                                    diff_html = ""
                                    status = "changed"
                            except Exception:
                                diff_html = ""
                                status = "changed"
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
    
    def _group_text_into_blocks(self, lines: List[str]) -> List[str]:
        """Group text lines into logical blocks for semantic comparison."""
        blocks = []
        current_block = []
        
        for line in lines:
            # Check if this line could be part of current block
            if not line.strip():
                # Empty line - finish current block if it exists
                if current_block:
                    blocks.append(" ".join(current_block))
                    current_block = []
                continue
                
            # Check if line ends with sentence-ending punctuation
            ends_sentence = bool(re.search(r'[.!?]"?\s*$', line))
            
            # Add line to current block
            current_block.append(line)
            
            # If line ends a sentence and block is getting long, finish block
            if ends_sentence and len(" ".join(current_block)) > 50:
                blocks.append(" ".join(current_block))
                current_block = []
        
        # Add any remaining block
        if current_block:
            blocks.append(" ".join(current_block))
            
        return blocks
    
    def _match_text_blocks(self, blocks1, blocks2, embeddings1, embeddings2):
        """Match text blocks using semantic similarity."""
        matches = []
        
        # Compute similarity matrix
        similarity_matrix = np.inner(embeddings1, embeddings2)
        
        # Find best matches above threshold
        matched_j = set()
        
        for i in range(len(blocks1)):
            best_j = -1
            best_sim = self.semantic_similarity_threshold
            
            for j in range(len(blocks2)):
                if j in matched_j:
                    continue
                
                sim = similarity_matrix[i, j]
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            
            if best_j >= 0:
                matches.append((i, best_j, float(similarity_matrix[i, best_j])))
                matched_j.add(best_j)
        
        return matches
    
    def _align_text_based_on_blocks(self, lines1, lines2, blocks1, blocks2, block_matches):
        """Align text lines based on block matches."""
        # Map lines to their block index
        line_to_block1 = {}
        line_to_block2 = {}
        block_to_lines1 = defaultdict(list)
        block_to_lines2 = defaultdict(list)
        
        # Process first document
        line_idx = 0
        for block_idx, block in enumerate(blocks1):
            block_line_count = len(block.split('\n')) + block.count('. ') + 1
            block_line_count = max(1, min(block_line_count, 10))  # Sanity check
            
            for _ in range(block_line_count):
                if line_idx < len(lines1):
                    line_to_block1[line_idx] = block_idx
                    block_to_lines1[block_idx].append(line_idx)
                    line_idx += 1
                    
        # Process second document
        line_idx = 0
        for block_idx, block in enumerate(blocks2):
            block_line_count = len(block.split('\n')) + block.count('. ') + 1
            block_line_count = max(1, min(block_line_count, 10))  # Sanity check
            
            for _ in range(block_line_count):
                if line_idx < len(lines2):
                    line_to_block2[line_idx] = block_idx
                    block_to_lines2[block_idx].append(line_idx)
                    line_idx += 1
        
        # Create new line arrays with aligned content
        aligned_lines1 = lines1.copy()
        aligned_lines2 = lines2.copy()
        
        # Cannot reliably align without block matches
        if not block_matches:
            return aligned_lines1, aligned_lines2
        
        # Use block matches to create an alignment mapping
        block_alignment = {}
        for i, j, sim in block_matches:
            block_alignment[i] = j
            
        # Create a line alignment based on block matches
        line_alignment = {}
        for line1_idx, block1_idx in line_to_block1.items():
            if block1_idx in block_alignment:
                block2_idx = block_alignment[block1_idx]
                lines2_in_block = block_to_lines2[block2_idx]
                
                # Pick corresponding line in block2
                if lines2_in_block:
                    rel_pos = line1_idx / max(1, len(block_to_lines1[block1_idx]))
                    line2_idx = lines2_in_block[min(int(rel_pos * len(lines2_in_block)), len(lines2_in_block) - 1)]
                    line_alignment[line1_idx] = line2_idx
        
        return aligned_lines1, aligned_lines2
    
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
        """
        Generate enhanced HTML showing differences between two tables with 
        column-level highlighting and improved status indication.
        """
        max_rows = max(len(table1), len(table2))
        max_cols = max(
            max((len(row) for row in table1), default=0),
            max((len(row) for row in table2), default=0)
        )
        
        # Create enhanced side-by-side table diff
        html = ['<div class="table-diff-container">']
        
        # Table 1 (Left side)
        html.append('<div class="table-diff-left">')
        html.append(f'<table class="diff-table source1" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i in range(max_rows):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Get row or empty list
            row = table1[i] if i < len(table1) else []
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(cell)
                
                # Check if this cell exists in table2
                cell_class = "empty" if not cell.strip() else ""
                cell_status = self._get_cell_status(i, j, table1, table2)
                
                if cell_status:
                    cell_class += f" {cell_status}"
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        
        # Table 2 (Right side)
        html.append('<div class="table-diff-right">')
        html.append(f'<table class="diff-table source2" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i in range(max_rows):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Get row or empty list
            row = table2[i] if i < len(table2) else []
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(cell)
                
                # Check if this cell exists in table1
                cell_class = "empty" if not cell.strip() else ""
                cell_status = self._get_cell_status(i, j, table2, table1, reverse=True)
                
                if cell_status:
                    cell_class += f" {cell_status}"
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        
        html.append('</div>')
        
        return ''.join(html)
        
        # Table 1 (Left side)
        html.append('<div class="table-diff-left">')
        html.append(f'<table class="diff-table source1" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i in range(max_rows):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Get row or empty list
            row = table1[i] if i < len(table1) else []
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(cell)
                
                # Check if this cell exists in table2
                cell_class = "empty" if not cell.strip() else ""
                cell_status = self._get_cell_status(i, j, table1, table2)
                
                if cell_status:
                    cell_class += f" {cell_status}"
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        
        # Table 2 (Right side)
        html.append('<div class="table-diff-right">')
        html.append(f'<table class="diff-table source2" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(max_cols):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i in range(max_rows):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Get row or empty list
            row = table2[i] if i < len(table2) else []
            
            # Fill cells
            for j in range(max_cols):
                cell = row[j] if j < len(row) else ""
                cell_html = self._escape_html(cell)
                
                # Check if this cell exists in table1
                cell_class = "empty" if not cell.strip() else ""
                cell_status = self._get_cell_status(i, j, table2, table1, reverse=True)
                
                if cell_status:
                    cell_class += f" {cell_status}"
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        
        html.append('</div>')
        
        return ''.join(html)
    
    def _get_cell_status(self, row, col, table1, table2, reverse=False):
        """
        Determine the status of a cell for HTML diff.
        
        Args:
            row: Row index
            col: Column index
            table1: Source table
            table2: Target table
            reverse: If True, consider table2->table1 instead of table1->table2
            
        Returns:
            Status class name or None
        """
        # Check if the cell exists in both tables
        cell1 = table1[row][col] if row < len(table1) and col < len(table1[row]) else ""
        cell2 = table2[row][col] if row < len(table2) and col < len(table2[row]) else ""
        
        cell1 = self._normalize_cell(cell1)
        cell2 = self._normalize_cell(cell2)
        
        # Empty cells don't need status
        if not cell1 and not cell2:
            return None
            
        # Cell only in table1
        if cell1 and not cell2:
            return "deleted" if not reverse else "inserted"
            
        # Cell only in table2
        if not cell1 and cell2:
            return "inserted" if not reverse else "deleted"
            
        # Both cells have content
        if cell1 == cell2:
            return "similar"  # Exact match
            
        # Check for fuzzy match
        if self._fuzzy_match_cells(cell1, cell2):
            return "similar"  # Similar content
            
        # Different content
        return "modified" 
    
    def _format_table_as_deleted(self, table_content: List[List[str]]) -> str:
        """Format a table that only exists in the first document."""
        if not table_content:
            return ""
            
        html = ['<div class="table-diff-container">']
        html.append('<div class="table-diff-left" style="width:100%">') 
        html.append('<table class="diff-table deleted" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(len(table_content[0])):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Fill cells
            for j, cell in enumerate(row):
                cell_html = self._escape_html(cell)
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
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(len(table_content[0])):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Fill cells
            for j, cell in enumerate(row):
                cell_html = self._escape_html(cell)
                cell_class = "inserted"  # All cells are inserted
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        html.append('</div>')
        
        return ''.join(html)
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters in text."""
        if not text:
            return ""
            
        return (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))
                   
    def _format_table_as_deleted(self, table_content: List[List[str]]) -> str:
        """Format a table that only exists in the first document."""
        if not table_content:
            return ""
            
        html = ['<div class="table-diff-container">']
        html.append('<div class="table-diff-left" style="width:100%">') 
        html.append('<table class="diff-table deleted" cellspacing="0" cellpadding="3">')
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(len(table_content[0]) if table_content and table_content[0] else 0):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Fill cells
            for j, cell in enumerate(row):
                cell_html = self._escape_html(cell)
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
        
        # Column headers
        html.append('<tr class="column-headers">')
        html.append('<th class="row-index-header">#</th>')  # Row index header
        for j in range(len(table_content[0]) if table_content and table_content[0] else 0):
            html.append(f'<th>Col {j+1}</th>')
        html.append('</tr>')
        
        # Table rows
        for i, row in enumerate(table_content):
            row_class = "header-row" if i == 0 else ""
            html.append(f'<tr class="{row_class}" data-row="{i}">')
            html.append(f'<td class="row-index">{i+1}</td>')  # Row index
            
            # Fill cells
            for j, cell in enumerate(row):
                cell_html = self._escape_html(cell)
                cell_class = "inserted"  # All cells are inserted
                    
                html.append(f'<td class="{cell_class}" data-row="{i}" data-col="{j}">{cell_html}</td>')
            
            html.append('</tr>')
        
        html.append('</table>')
        html.append('</div>')
        html.append('</div>')
        
        return ''.join(html)
        
    def _get_cell_status(self, row, col, table1, table2, reverse=False):
        """
        Determine the status of a cell for HTML diff.
        
        Args:
            row: Row index
            col: Column index
            table1: Source table
            table2: Target table
            reverse: If True, consider table2->table1 instead of table1->table2
            
        Returns:
            Status class name or None
        """
        # Check if the cell exists in both tables
        cell1 = table1[row][col] if row < len(table1) and col < len(table1[row]) else ""
        cell2 = table2[row][col] if row < len(table2) and col < len(table2[row]) else ""
        
        cell1 = self._normalize_cell(cell1)
        cell2 = self._normalize_cell(cell2)
        
        # Empty cells don't need status
        if not cell1 and not cell2:
            return None
            
        # Cell only in table1
        if cell1 and not cell2:
            return "deleted" if not reverse else "inserted"
            
        # Cell only in table2
        if not cell1 and cell2:
            return "inserted" if not reverse else "deleted"
            
        # Both cells have content
        if cell1 == cell2:
            return "similar"  # Exact match
            
        # Check for fuzzy match
        if self._fuzzy_match_cells(cell1, cell2):
            return "similar"  # Similar content
            
        # Different content
        return "modified"
