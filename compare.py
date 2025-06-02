import logging
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import math # For more advanced similarity thresholds

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PdfCompare:
    """
    Enhanced PDF comparator with improved table detection and text difference analysis.
    Includes optimizations for performance and accuracy.
    """

    def __init__(self,
                 diff_threshold: float = 0.75,
                 cell_match_threshold: float = 0.9,
                 fuzzy_match_threshold: float = 0.8,
                 max_workers: int = None):
        """
        Initialize the PDF comparator with customizable thresholds.
        """
        self.diff_threshold = diff_threshold
        self.cell_match_threshold = cell_match_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.max_workers = max_workers if max_workers is not None else (os.cpu_count() or 4)
        if self.max_workers <= 0: # Ensure at least 1 worker for sequential execution if 0 or negative is passed
            self.max_workers = 1 
            self.executor = None # Or handle single-threaded execution explicitly
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.table_comparison_cache = {}
        self.text_similarity_cache = {}
        
        logger.info(f"PdfCompare initialized with: diff_threshold={diff_threshold}, "
                    f"cell_match_threshold={cell_match_threshold}, "
                    f"fuzzy_match_threshold={fuzzy_match_threshold}, max_workers={self.max_workers}")

    def _execute_parallel(self, func, items, *args, **kwargs):
        """Helper to run functions in parallel or sequentially if executor is not available."""
        if self.executor and self.max_workers > 1: # Only parallelize if more than 1 worker
            futures = [self.executor.submit(func, item, *args, **kwargs) for item in items]
            return [future.result() for future in as_completed(futures)]
        else:
            return [func(item, *args, **kwargs) for item in items]

    def compare_pdfs(self, pdf1: Dict, pdf2: Dict,
                     progress_callback: Callable = None) -> Dict:
        """
        Compare two PDFs and generate detailed differences.
        """
        logger.info("Starting PDF comparison.")
        max_pages = max(len(pdf1), len(pdf2))
        results = {"max_pages": max_pages, "pages": {}}

        if progress_callback: progress_callback(0.1)

        # Optimization 1: Collect tables with pre-computed hash for raw content
        tables1 = list(self._collect_tables(pdf1))
        tables2 = list(self._collect_tables(pdf2))
        text_blocks1 = self._collect_text_blocks(pdf1)
        text_blocks2 = self._collect_text_blocks(pdf2)

        logger.info(f"PDF1: {len(text_blocks1)} text blocks, {len(tables1)} tables.")
        logger.info(f"PDF2: {len(text_blocks2)} text blocks, {len(tables2)} tables.")
        if progress_callback: progress_callback(0.2)

        # Match tables globally
        table_matches = self._match_tables_global(tables1, tables2)
        if progress_callback: progress_callback(0.5)

        # Match text blocks globally
        text_matches_raw = self._match_text_blocks_global(text_blocks1, text_blocks2)
        if progress_callback: progress_callback(0.7)
        
        # Post-process text matches (e.g., for splits, continuations)
        text_matches_processed = self._post_process_text_matches(text_matches_raw)
        if progress_callback: progress_callback(0.8)

        # Organize matches by page for the report
        organized_diffs = self._organize_differences_by_page(text_matches_processed, table_matches)
        
        for p in range(1, max_pages + 1):
            results["pages"][p] = organized_diffs.get(p, {
                "text_differences": [],
                "table_differences": []
            })

        if progress_callback: progress_callback(1.0)
        logger.info("PDF comparison completed.")
        return results

    def _collect_text_blocks(self, pdf_data: Dict) -> List[Dict]:
        """Extract text blocks with page info, bbox, and hash."""
        text_blocks = []
        for pg, pdata in pdf_data.items():
            for idx, elem in enumerate(pdata.get("elements", [])):
                if elem.get("type") == "text":
                    content = elem.get("content", "").strip()
                    if content:
                        text_blocks.append({
                            "page": pg,
                            "content": content,
                            "index": idx, # Original index on page
                            "bbox": elem["bbox"],
                            "hash": hashlib.md5(content.encode()).hexdigest(),
                            "line_count": content.count('\n') + 1,
                            "word_count": len(content.split())
                        })
        return text_blocks

    def _collect_tables(self, pdf_data: Dict) -> List[Tuple[int, Dict]]:
        """Extract tables with page info and pre-compute content hash."""
        tables = []
        for pg, pdata in pdf_data.items():
            for elem in pdata.get("elements", []):
                if elem.get("type") == "table":
                    # Optimization 5: Pre-compute content hash for table
                    table_content_str = str(elem.get("content"))
                    elem["content_hash"] = hashlib.md5(table_content_str.encode()).hexdigest()
                    tables.append((pg, elem))
        return tables

    def _match_text_blocks_global(self, blocks1: List[Dict], blocks2: List[Dict]) -> List[Dict]:
        """Match text blocks globally, prioritizing exact matches, then fuzzy."""
        matches = []
        
        blocks2_by_hash = defaultdict(list)
        for block in blocks2:
            blocks2_by_hash[block["hash"]].append(block)

        matched_block1_indices = set()
        matched_block2_indices = set()

        # Pass 1: Exact matches (by hash)
        for i, b1 in enumerate(blocks1):
            if b1["hash"] in blocks2_by_hash:
                for b2_candidate in blocks2_by_hash[b1["hash"]]:
                    # Find original index for tracking, assumes blocks2 is stable
                    try:
                        b2_idx = next(idx for idx, block in enumerate(blocks2) if block is b2_candidate)
                    except StopIteration:
                        continue # Should not happen if b2_candidate came from blocks2

                    if b2_idx not in matched_block2_indices:
                        status = "moved" if b1["page"] != b2_candidate["page"] else "matched"
                        matches.append({
                            "type": "text",
                            "block1": b1, "block2": b2_candidate, "status": status, "score": 1.0,
                            "text1": b1["content"], "text2": b2_candidate["content"],
                            "page1": b1["page"], "page2": b2_candidate["page"]
                        })
                        matched_block1_indices.add(i)
                        matched_block2_indices.add(b2_idx)
                        break 

        blocks1_remaining = [(i, b) for i, b in enumerate(blocks1) if i not in matched_block1_indices]
        blocks2_remaining = [(i, b) for i, b in enumerate(blocks2) if i not in matched_block2_indices]
        
        # Pass 2: Fuzzy matches for remaining blocks - Parallelized
        # Optimization 1: Generate tasks for parallel fuzzy matching
        match_tasks = []
        for b1_idx, b1 in blocks1_remaining:
            for b2_idx, b2 in blocks2_remaining:
                match_tasks.append(((b1_idx, b1), (b2_idx, b2)))

        # Use _execute_parallel for similarity calculations
        # Need a helper function to wrap the similarity calculation and return the full match info
        def _calculate_fuzzy_match_score(task_tuple):
            (b1_idx, b1), (b2_idx, b2) = task_tuple
            text1_content = b1.get("content", "")
            text2_content = b2.get("content", "")
            cache_key_tuple = tuple(sorted((b1["hash"], b2["hash"])))

            if cache_key_tuple in self.text_similarity_cache:
                score = self.text_similarity_cache[cache_key_tuple]
            else:
                score = self._calculate_text_similarity(text1_content, text2_content)
                self.text_similarity_cache[cache_key_tuple] = score
            
            return {
                "b1_idx": b1_idx, "b1": b1,
                "b2_idx": b2_idx, "b2": b2,
                "score": score
            }

        if match_tasks:
            logger.info(f"Executing {len(match_tasks)} fuzzy text match tasks in parallel.")
            # Use a smaller chunk size for results if memory is an issue for very large lists
            all_scores = self._execute_parallel(_calculate_fuzzy_match_score, match_tasks)
            logger.info("Finished fuzzy text match tasks.")

            # Process results to find best matches, resolving conflicts greedily
            # Sort by score descending
            all_scores.sort(key=lambda x: x["score"], reverse=True)

            current_matched_b1_indices = set()
            current_matched_b2_indices = set()

            for result in all_scores:
                b1_idx = result["b1_idx"]
                b2_idx = result["b2_idx"]
                score = result["score"]

                if score < self.fuzzy_match_threshold:
                    continue

                if b1_idx not in current_matched_b1_indices and b2_idx not in current_matched_b2_indices:
                    b1 = result["b1"]
                    b2 = result["b2"]
                    status = "moved" if b1["page"] != b2["page"] else "modified"
                    matches.append({
                        "type": "text",
                        "block1": b1, "block2": b2, "status": status, "score": score,
                        "text1": b1["content"], "text2": b2["content"],
                        "page1": b1["page"], "page2": b2["page"]
                    })
                    current_matched_b1_indices.add(b1_idx)
                    current_matched_b2_indices.add(b2_idx)
        
        # Add unmatched blocks from pdf1 (deleted)
        for i, b1 in blocks1_remaining:
            if i not in matched_block1_indices and i not in current_matched_b1_indices:
                matches.append({
                    "type": "text",
                    "block1": b1, "block2": None, "status": "deleted", "score": 0.0,
                    "text1": b1["content"], "text2": "",
                    "page1": b1["page"], "page2": None
                })

        # Add unmatched blocks from pdf2 (inserted)
        for i, b2 in blocks2_remaining:
            if i not in matched_block2_indices and i not in current_matched_b2_indices:
                matches.append({
                    "type": "text",
                    "block1": None, "block2": b2, "status": "inserted", "score": 0.0,
                    "text1": "", "text2": b2["content"],
                    "page1": None, "page2": b2["page"]
                })
        return matches

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 and not text2: return 1.0
        if not text1 or not text2: return 0.0
        return SequenceMatcher(None, text1, text2).ratio()

    def _post_process_text_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Post-process text matches. Currently a placeholder.
        """
        logger.debug(f"Post-processing {len(matches)} text matches. Currently a passthrough.")
        return matches


    def _match_tables_global(self, tables1_orig: List[Tuple[int, Dict]], tables2_orig: List[Tuple[int, Dict]]) -> List[Dict]:
        """Match tables globally using content hash and similarity."""
        matches = []
        
        # Decorate tables with original index and page for easier tracking
        tables1 = [(pg, tbl, idx) for idx, (pg, tbl) in enumerate(tables1_orig)]
        tables2 = [(pg, tbl, idx) for idx, (pg, tbl) in enumerate(tables2_orig)]

        tables2_by_hash = defaultdict(list)
        for pg, tbl, idx in tables2: # This loop expects (pg, tbl, idx)
            if "content_hash" in tbl: # Should always be present due to _collect_tables optimization
                tables2_by_hash[tbl["content_hash"]].append((pg, tbl, idx))
        
        matched_t1_indices_exact = set()
        matched_t2_indices_exact = set()

        # Pass 1: Exact matches by content_hash
        for pg1, t1, t1_idx in tables1: # This loop expects (pg, t1, t1_idx)
            if t1.get("content_hash") in tables2_by_hash:
                for pg2_cand, t2_cand, t2_idx_cand in tables2_by_hash[t1["content_hash"]]:
                    if t2_idx_cand not in matched_t2_indices_exact:
                        status = "moved" if pg1 != pg2_cand else "matched"
                        # For exact matches, diff_details can be generated here or deferred
                        diff_details = self._compare_individual_tables(t1, t2_cand)
                        matches.append({
                            "type": "table", "status": status, "score": 1.0,
                            "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                            "page2": pg2_cand, "table2_content": t2_cand.get("content"), "table2_bbox": t2_cand.get("bbox"), "table2_id": t2_cand.get("table_id"),
                            "diff_html": diff_details["html"]
                        })
                        matched_t1_indices_exact.add(t1_idx)
                        matched_t2_indices_exact.add(t2_idx_cand)
                        break 
        
        # FIX for ValueError: too many values to unpack (expected 2)
        # tables1_remaining and tables2_remaining are already lists of 3-tuples (pg, t, idx)
        tables1_remaining = [(pg, t, idx) for pg, t, idx in tables1 if idx not in matched_t1_indices_exact]
        tables2_remaining = [(pg, t, idx) for pg, t, idx in tables2 if idx not in matched_t2_indices_exact]

        # Pass 2: Similarity matches for remaining tables - Parallelized
        # Optimization 1: Generate tasks for parallel fuzzy matching
        table_match_tasks = []
        # FIX for ValueError: too many values to unpack (expected 2)
        # Directly unpack the 3-tuple (pg1, t1, t1_idx) as tables1_remaining elements are already 3-tuples
        for pg1, t1, t1_idx in tables1_remaining:
            # Directly unpack the 3-tuple (pg2, t2, t2_idx) as tables2_remaining elements are already 3-tuples
            for pg2, t2, t2_idx in tables2_remaining:
                table_match_tasks.append(((pg1, t1, t1_idx), (pg2, t2, t2_idx)))

        def _calculate_table_fuzzy_match_score(task_tuple):
            (pg1, t1, t1_idx), (pg2, t2, t2_idx) = task_tuple
            
            if not isinstance(t1, dict) or not isinstance(t2, dict):
                # This case should ideally not happen if tables1/tables2 are correctly populated
                return {"t1_info": (pg1, t1, t1_idx), "t2_info": (pg2, t2, t2_idx), "score": 0.0}

            score = self._calculate_table_similarity(t1, t2)
            return {
                "t1_info": (pg1, t1, t1_idx),
                "t2_info": (pg2, t2, t2_idx),
                "score": score
            }

        if table_match_tasks:
            logger.info(f"Executing {len(table_match_tasks)} fuzzy table match tasks in parallel.")
            all_table_scores = self._execute_parallel(_calculate_table_fuzzy_match_score, table_match_tasks)
            logger.info("Finished fuzzy table match tasks.")

            all_table_scores.sort(key=lambda x: x["score"], reverse=True)
            
            current_matched_t1_indices = set()
            current_matched_t2_indices = set()

            for result in all_table_scores:
                score = result["score"]
                pg1, t1, t1_idx = result["t1_info"]
                pg2, t2, t2_idx = result["t2_info"]

                if score < self.diff_threshold:
                    continue

                if (t1_idx not in matched_t1_indices_exact and t1_idx not in current_matched_t1_indices) and \
                   (t2_idx not in matched_t2_indices_exact and t2_idx not in current_matched_t2_indices):
                    
                    status = "moved" if pg1 != pg2 else "modified"
                    diff_details = self._compare_individual_tables(t1, t2)
                    matches.append({
                        "type": "table", "status": status, "score": score,
                        "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                        "page2": pg2, "table2_content": t2.get("content"), "table2_bbox": t2.get("bbox"), "table2_id": t2.get("table_id"),
                        "diff_html": diff_details["html"]
                    })
                    current_matched_t1_indices.add(t1_idx)
                    current_matched_t2_indices.add(t2_idx)

        # Add tables only in pdf1 (deleted)
        # FIX: Directly unpack the 3-tuple (pg1, t1, t1_idx)
        for pg1, t1, t1_idx in tables1_remaining:
            if t1_idx not in matched_t1_indices_exact and t1_idx not in current_matched_t1_indices:
                diff_details = self._compare_individual_tables(t1, None)
                matches.append({
                    "type": "table", "status": "deleted", "score": 0.0,
                    "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                    "page2": None, "table2_content": None, "table2_bbox": None, "table2_id": None,
                    "diff_html": diff_details["html"]
                })

        # Add tables only in pdf2 (inserted)
        # FIX: Directly unpack the 3-tuple (pg2, t2, t2_idx)
        for pg2, t2, t2_idx in tables2_remaining:
            if t2_idx not in matched_t2_indices_exact and t2_idx not in current_matched_t2_indices:
                diff_details = self._compare_individual_tables(None, t2)
                matches.append({
                    "type": "table", "status": "inserted", "score": 0.0,
                    "page1": None, "table1_content": None, "table1_bbox": None, "table1_id": None,
                    "page2": pg2, "table2_content": t2.get("content"), "table2_bbox": t2.get("bbox"), "table2_id": t2.get("table_id"),
                    "diff_html": diff_details["html"]
                })
        return matches

    def _calculate_table_similarity(self, t1: Dict, t2: Dict) -> float:
        """Calculate similarity between two tables based on content and structure."""
        if not t1 or not t2 or not t1.get("content") or not t2.get("content"):
            return 0.0
        
        # Use content hash for quick check (already pre-computed)
        if t1.get("content_hash") and t2.get("content_hash") and t1["content_hash"] == t2["content_hash"]:
            return 1.0

        # Cache key using sorted hashes of table content strings
        content1_str = str(t1.get("content"))
        content2_str = str(t2.get("content"))
        # Using pre-computed hash if available, otherwise compute it
        hash1 = t1.get("content_hash", hashlib.md5(content1_str.encode()).hexdigest())
        hash2 = t2.get("content_hash", hashlib.md5(content2_str.encode()).hexdigest())
        
        cache_key_tuple = tuple(sorted((hash1, hash2)))

        if cache_key_tuple in self.table_comparison_cache:
            return self.table_comparison_cache[cache_key_tuple]

        struct_sim = self._table_structure_similarity(t1.get("content"), t2.get("content"))
        content_sim = self._table_cell_content_similarity(t1.get("content"), t2.get("content"))
        
        similarity = (struct_sim * 0.4) + (content_sim * 0.6)
        self.table_comparison_cache[cache_key_tuple] = similarity
        return similarity

    def _table_structure_similarity(self, c1: List[List[str]], c2: List[List[str]]) -> float:
        """Compare row and column counts."""
        rows1, cols1 = (len(c1), len(c1[0]) if c1 else 0)
        rows2, cols2 = (len(c2), len(c2[0]) if c2 else 0)
        if rows1 == 0 and rows2 == 0: return 1.0
        if rows1 == 0 or rows2 == 0: return 0.0
        
        row_sim = min(rows1, rows2) / max(rows1, rows2)
        col_sim = min(cols1, cols2) / max(cols1, cols2) if max(cols1,cols2) > 0 else 0
        return (row_sim + col_sim) / 2

    def _table_cell_content_similarity(self, c1: List[List[str]], c2: List[List[str]]) -> float:
        """Compare cell content using SequenceMatcher."""
        flat_text1 = " ".join([" ".join(map(str, row)) for row in c1])
        flat_text2 = " ".join([" ".join(map(str, row)) for row in c2])
        return self._calculate_text_similarity(flat_text1, flat_text2)

    def _compare_individual_tables(self, table1_data: Optional[Dict], table2_data: Optional[Dict]) -> Dict:
        """
        Generates HTML for side-by-side comparison of two tables, or shows one as deleted/inserted.
        """
        content1 = table1_data.get("content") if table1_data else []
        content2 = table2_data.get("content") if table2_data else []

        html = ['<div class="table-diff-container" style="display: flex; width: 100%;">']
        
        # Table 1 (Left side)
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        if content1:
            html.append(f'<h6>{self._escape_html(table1_data.get("table_id", "Table from PDF 1"))} (PDF 1)</h6>')
            html.append(self._render_html_table(content1, content2, is_left=True))
        else:
            html.append("<p>Not present in PDF 1</p>")
        html.append('</div>')

        # Separator
        html.append('<div style="width: 1px; background-color: #ccc;"></div>')

        # Table 2 (Right side)
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        if content2:
            html.append(f'<h6>{self._escape_html(table2_data.get("table_id", "Table from PDF 2"))} (PDF 2)</h6>')
            html.append(self._render_html_table(content2, content1, is_left=False))
        else:
            html.append("<p>Not present in PDF 2</p>")
        html.append('</div>')
        
        html.append('</div>')
        return {"html": "".join(html)}

    def _render_html_table(self, main_content: List[List[str]], 
                           compare_content: Optional[List[List[str]]], 
                           is_left: bool) -> str:
        """Renders a single table into HTML, highlighting differences against compare_content."""
        if not main_content:
            return "<p>Table is empty or not provided.</p>"

        html = ['<table class="diff-table" border="1" style="border-collapse: collapse; width: 100%;">']
        
        num_main_rows = len(main_content)
        num_main_cols = max(len(row) for row in main_content) if main_content else 0
        
        num_compare_rows = len(compare_content) if compare_content else 0
        num_compare_cols = max(len(row) for row in compare_content) if compare_content and any(compare_content) else 0

        max_rows = max(num_main_rows, num_compare_rows)
        max_cols = max(num_main_cols, num_compare_cols)

        for i in range(max_rows):
            html.append("<tr>")
            for j in range(max_cols):
                cell_main_text = ""
                cell_compare_text = ""
                cell_class = ""

                if i < num_main_rows and j < num_main_cols:
                    cell_main_text = str(main_content[i][j])
                
                if compare_content and i < num_compare_rows and j < num_compare_cols:
                    cell_compare_text = str(compare_content[i][j])

                # Determine cell class
                if is_left: # Comparing main_content (pdf1) to compare_content (pdf2)
                    if i < num_main_rows and j < num_main_cols: # Cell exists in main
                        if compare_content and i < num_compare_rows and j < num_compare_cols: # Cell also exists in compare
                            if self._normalize_cell(cell_main_text) == self._normalize_cell(cell_compare_text):
                                cell_class = "cell-similar"
                            else:
                                cell_class = "cell-modified"
                        else: # Cell only in main (deleted from compare's perspective)
                            cell_class = "cell-deleted"
                    else: # Cell placeholder for alignment, doesn't exist in main
                        cell_class = "cell-inserted" # (inserted in compare) - should be styled as empty on this side
                        cell_main_text = "&nbsp;" # Placeholder
                else: # Comparing main_content (pdf2) to compare_content (pdf1)
                    if i < num_main_rows and j < num_main_cols: # Cell exists in main (pdf2)
                        if compare_content and i < num_compare_rows and j < num_compare_cols: # Cell also exists in compare (pdf1)
                            if self._normalize_cell(cell_main_text) == self._normalize_cell(cell_compare_text):
                                cell_class = "cell-similar"
                            else:
                                cell_class = "cell-modified"
                        else: # Cell only in main (pdf2) (inserted from pdf1's perspective)
                            cell_class = "cell-inserted"
                    else: # Cell placeholder for alignment
                        cell_class = "cell-deleted" # (deleted from pdf1) - should be styled as empty on this side
                        cell_main_text = "&nbsp;" # Placeholder
                
                html.append(f'<td class="{cell_class}">{self._escape_html(cell_main_text)}</td>')
            html.append("</tr>")
        html.append("</table>")
        return "".join(html)

    def _normalize_cell(self, cell_text: str) -> str:
        """Normalize cell text for comparison."""
        return str(cell_text).strip().lower()

    def _escape_html(self, text: str) -> str:
        """Basic HTML escaping."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _organize_differences_by_page(self, text_matches: List[Dict], table_matches: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
        """
        Organizes all differences (text and table) by page.
        Ensures that "moved" content (both text and tables) is correctly represented
        for side-by-side display in the report.
        """
        organized_diffs = defaultdict(lambda: {"text_differences": [], "table_differences": []})
        
        # Optimization 6: Use existing hashes from block objects
        moved_text_fingerprints_pdf1 = set() 
        moved_text_fingerprints_pdf2 = set()

        for match in text_matches:
            if match["status"] == "moved":
                # Ensure block1 and block2 exist before trying to access their 'hash'
                if match.get("block1") and "hash" in match["block1"]:
                    moved_text_fingerprints_pdf1.add(match["block1"]["hash"])
                if match.get("block2") and "hash" in match["block2"]:
                     moved_text_fingerprints_pdf2.add(match["block2"]["hash"])
        
        for match in text_matches:
            status = match["status"]
            
            # FIX for AttributeError: 'NoneType' object has no attribute 'get'
            # Check if block1 exists before calling .get("hash")
            text1_hash = match.get("block1").get("hash") if match.get("block1") else None
            # Check if block2 exists before calling .get("hash")
            text2_hash = match.get("block2").get("hash") if match.get("block2") else None

            page1 = match.get("page1")
            page2 = match.get("page2")

            # Skip deleted/inserted if they are part of an already accounted "moved" operation
            if status == "deleted" and text1_hash is not None and text1_hash in moved_text_fingerprints_pdf1:
                continue
            if status == "inserted" and text2_hash is not None and text2_hash in moved_text_fingerprints_pdf2:
                continue

            diff_item = {
                "status": status, "score": match.get("score", 0.0),
                "text1": match.get("text1", ""), "text2": match.get("text2", ""),
                "page1": page1, "page2": page2,
                "block1_bbox": match.get("block1", {}).get("bbox") if match.get("block1") else None,
                "block2_bbox": match.get("block2", {}).get("bbox") if match.get("block2") else None,
            }

            if status == "moved":
                if page1 is not None:
                    organized_diffs[page1]["text_differences"].append(diff_item)
                if page2 is not None and page1 != page2:
                    organized_diffs[page2]["text_differences"].append(diff_item)
            elif status == "deleted" and page1 is not None:
                organized_diffs[page1]["text_differences"].append(diff_item)
            elif status == "inserted" and page2 is not None:
                organized_diffs[page2]["text_differences"].append(diff_item)
            elif status == "modified" and page1 is not None:
                 organized_diffs[page1]["text_differences"].append(diff_item)
            elif status == "matched" and match.get("score", 1.0) < 1.0 and page1 is not None:
                 organized_diffs[page1]["text_differences"].append(diff_item)

        # Process table matches
        moved_table_fingerprints_pdf1 = set()
        moved_table_fingerprints_pdf2 = set()

        for match in table_matches:
            if match["status"] == "moved":
                if match.get("table1_content"):
                    # Use the pre-computed hash if available from the table object
                    # Fallback to re-hashing if not explicitly stored in match (though it should be)
                    table1_hash = match.get("table1_content_hash", hashlib.md5(str(match["table1_content"]).encode()).hexdigest())
                    moved_table_fingerprints_pdf1.add(table1_hash)
                if match.get("table2_content"):
                    table2_hash = match.get("table2_content_hash", hashlib.md5(str(match["table2_content"]).encode()).hexdigest())
                    moved_table_fingerprints_pdf2.add(table2_hash)
        
        for match in table_matches:
            status = match["status"]
            page1 = match.get("page1")
            page2 = match.get("page2")
            table1_content = match.get("table1_content")
            table2_content = match.get("table2_content")

            # FIX: Ensure table1_content and table2_content are not None before hashing
            table1_hash = match.get("table1_content_hash", hashlib.md5(str(table1_content).encode()).hexdigest() if table1_content is not None else None)
            table2_hash = match.get("table2_content_hash", hashlib.md5(str(table2_content).encode()).hexdigest() if table2_content is not None else None)
            
            if status == "deleted" and table1_hash is not None and table1_hash in moved_table_fingerprints_pdf1:
                continue
            if status == "inserted" and table2_hash is not None and table2_hash in moved_table_fingerprints_pdf2:
                continue

            diff_item = {
                "status": status, "score": match.get("score", 0.0),
                "page1": page1, "page2": page2,
                "table1_id": match.get("table1_id"), "table2_id": match.get("table2_id"),
                "table1_bbox": match.get("table1_bbox"), "table2_bbox": match.get("table2_bbox"),
                "diff_html": match.get("diff_html", "")
            }

            if status == "moved":
                if page1 is not None:
                    organized_diffs[page1]["table_differences"].append(diff_item)
                if page2 is not None and page1 != page2:
                     organized_diffs[page2]["table_differences"].append(diff_item)
            elif status == "deleted" and page1 is not None:
                organized_diffs[page1]["table_differences"].append(diff_item)
            elif status == "inserted" and page2 is not None:
                organized_diffs[page2]["table_differences"].append(diff_item)
            elif status == "modified" and page1 is not None:
                organized_diffs[page1]["table_differences"].append(diff_item)
            elif status == "matched" and match.get("score", 1.0) < 1.0 and page1 is not None:
                 organized_diffs[page1]["table_differences"].append(diff_item)


        # Final sort for consistent output
        for page_num in organized_diffs:
            organized_diffs[page_num]["text_differences"].sort(key=lambda d: (
                d.get("block1_bbox")[1] if d.get("block1_bbox") else (d.get("block2_bbox")[1] if d.get("block2_bbox") else float('inf')),
                d.get("block1_bbox")[0] if d.get("block1_bbox") else (d.get("block2_bbox")[0] if d.get("block2_bbox") else float('inf'))
            ))
            organized_diffs[page_num]["table_differences"].sort(key=lambda d: (
                d.get("table1_bbox")[1] if d.get("table1_bbox") else (d.get("table2_bbox")[1] if d.get("table2_bbox") else float('inf')),
                d.get("table1_bbox")[0] if d.get("table1_bbox") else (d.get("table2_bbox")[0] if d.get("table2_bbox") else float('inf'))
            ))
            
        return organized_diffs
