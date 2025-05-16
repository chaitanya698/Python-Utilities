import logging
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict

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
        self.max_workers = max_workers or os.cpu_count() or 4 # Ensure at least 1 worker
        # Ensure executor is initialized only if max_workers > 0
        if self.max_workers > 0:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None # Or handle single-threaded execution

        self.table_comparison_cache = {}
        self.text_similarity_cache = {}
        
        logger.info(f"PdfCompare initialized with: diff_threshold={diff_threshold}, "
                    f"cell_match_threshold={cell_match_threshold}, "
                    f"fuzzy_match_threshold={fuzzy_match_threshold}, max_workers={self.max_workers}")

    def _execute_parallel(self, func, items):
        """Helper to run functions in parallel or sequentially if executor is not available."""
        if self.executor:
            futures = [self.executor.submit(func, item) for item in items]
            return [future.result() for future in as_completed(futures)]
        else:
            return [func(item) for item in items]

    def compare_pdfs(self, pdf1: Dict, pdf2: Dict,
                     progress_callback: Callable = None) -> Dict:
        """
        Compare two PDFs and generate detailed differences.
        """
        logger.info("Starting PDF comparison.")
        max_pages = max(len(pdf1), len(pdf2))
        results = {"max_pages": max_pages, "pages": {}}

        if progress_callback: progress_callback(0.1)

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
        # This step needs to be carefully reviewed if it alters text1/text2 incorrectly
        text_matches_processed = self._post_process_text_matches(text_matches_raw)
        if progress_callback: progress_callback(0.8)

        # Organize matches by page for the report
        # This is a critical step for correct "moved" handling
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
        """Extract tables with page info."""
        tables = []
        for pg, pdata in pdf_data.items():
            for elem in pdata.get("elements", []):
                if elem.get("type") == "table":
                    tables.append((pg, elem))
        return tables

    def _match_text_blocks_global(self, blocks1: List[Dict], blocks2: List[Dict]) -> List[Dict]:
        """Match text blocks globally, prioritizing exact matches, then fuzzy."""
        matches = []
        # Create a dictionary for quick lookup of blocks in pdf2 by hash
        blocks2_by_hash = defaultdict(list)
        for block in blocks2:
            blocks2_by_hash[block["hash"]].append(block)

        # Create a set of matched block2 indices to avoid reusing them
        matched_block2_indices = set()

        # Pass 1: Exact matches (by hash)
        for i, b1 in enumerate(blocks1):
            if b1["hash"] in blocks2_by_hash:
                # Find the best exact match (e.g., closest by original index if multiple)
                # For simplicity, take the first available one
                found_match_in_b2 = False
                for b2_candidate in blocks2_by_hash[b1["hash"]]:
                    b2_idx = blocks2.index(b2_candidate) # Get original index for tracking
                    if b2_idx not in matched_block2_indices:
                        status = "moved" if b1["page"] != b2_candidate["page"] else "matched"
                        matches.append({
                            "type": "text",
                            "block1": b1, "block2": b2_candidate, "status": status, "score": 1.0,
                            "text1": b1["content"], "text2": b2_candidate["content"],
                            "page1": b1["page"], "page2": b2_candidate["page"]
                        })
                        matched_block2_indices.add(b2_idx)
                        found_match_in_b2 = True
                        break 
                if found_match_in_b2:
                    blocks1[i] = None # Mark as matched

        blocks1_remaining = [b for b in blocks1 if b is not None]
        
        # Pass 2: Fuzzy matches for remaining blocks
        for b1 in blocks1_remaining:
            best_b2_match = None
            highest_score = self.fuzzy_match_threshold - 0.01 # Ensure score must be >= threshold

            for b2_idx, b2 in enumerate(blocks2):
                if b2_idx in matched_block2_indices:
                    continue

                # Ensure text1 and text2 are strings
                text1_content = b1.get("content", "")
                text2_content = b2.get("content", "")
                
                # Use a tuple of hashes for caching similarity to ensure order doesn't matter for cache key
                cache_key_tuple = tuple(sorted((b1["hash"], b2["hash"])))

                if cache_key_tuple in self.text_similarity_cache:
                    score = self.text_similarity_cache[cache_key_tuple]
                else:
                    score = self._calculate_text_similarity(text1_content, text2_content)
                    self.text_similarity_cache[cache_key_tuple] = score
                
                if score > highest_score:
                    highest_score = score
                    best_b2_match = b2
            
            if best_b2_match:
                b2_match_idx = blocks2.index(best_b2_match)
                status = "moved" if b1["page"] != best_b2_match["page"] else "modified"
                matches.append({
                    "type": "text",
                    "block1": b1, "block2": best_b2_match, "status": status, "score": highest_score,
                    "text1": b1["content"], "text2": best_b2_match["content"], # Ensure text2 is correctly populated
                    "page1": b1["page"], "page2": best_b2_match["page"]
                })
                matched_block2_indices.add(b2_match_idx)
            else: # b1 is in pdf1 only (deleted)
                matches.append({
                    "type": "text",
                    "block1": b1, "block2": None, "status": "deleted", "score": 0.0,
                    "text1": b1["content"], "text2": "",
                    "page1": b1["page"], "page2": None
                })

        # Add blocks that are in pdf2 only (inserted)
        for b2_idx, b2 in enumerate(blocks2):
            if b2_idx not in matched_block2_indices:
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
        Post-process text matches. For now, this is a placeholder.
        Complex logic for splits/continuations should be carefully implemented
        to ensure text1 and text2 remain accurate representations of the original blocks.
        """
        # Example: could try to merge adjacent "modified" or "moved" blocks if they form a logical unit.
        # For now, return as is to avoid introducing errors in text1/text2.
        logger.debug(f"Post-processing {len(matches)} text matches. Currently a passthrough.")
        return matches


    def _match_tables_global(self, tables1_orig: List[Tuple[int, Dict]], tables2_orig: List[Tuple[int, Dict]]) -> List[Dict]:
        """Match tables globally using content hash and similarity."""
        matches = []
        
        # Decorate tables with original index and page for easier tracking
        tables1 = [(pg, tbl, idx) for idx, (pg, tbl) in enumerate(tables1_orig)]
        tables2 = [(pg, tbl, idx) for idx, (pg, tbl) in enumerate(tables2_orig)]

        tables2_by_hash = defaultdict(list)
        for pg, tbl, idx in tables2:
            if "content_hash" in tbl:
                tables2_by_hash[tbl["content_hash"]].append((pg, tbl, idx))
        
        matched_t2_indices = set()

        # Pass 1: Exact matches by content_hash
        for pg1, t1, t1_idx in tables1:
            if t1.get("content_hash") in tables2_by_hash:
                for pg2_cand, t2_cand, t2_idx_cand in tables2_by_hash[t1["content_hash"]]:
                    if t2_idx_cand not in matched_t2_indices:
                        status = "moved" if pg1 != pg2_cand else "matched"
                        diff_details = self._compare_individual_tables(t1, t2_cand)
                        matches.append({
                            "type": "table", "status": status, "score": 1.0,
                            "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                            "page2": pg2_cand, "table2_content": t2_cand.get("content"), "table2_bbox": t2_cand.get("bbox"), "table2_id": t2_cand.get("table_id"),
                            "diff_html": diff_details["html"]
                        })
                        matched_t2_indices.add(t2_idx_cand)
                        tables1[t1_idx] = None # Mark as matched
                        break 
        
        tables1_remaining = [t for t in tables1 if t is not None]

        # Pass 2: Similarity matches for remaining tables
        # This part can be computationally expensive and could be parallelized
        # For simplicity in this refactor, keeping it sequential.
        # Consider using self._execute_parallel if performance is an issue.

        temp_similarity_matches = []
        for pg1, t1, t1_idx in tables1_remaining:
            best_t2_match_info = None
            highest_score = self.diff_threshold - 0.01

            for pg2, t2, t2_idx in tables2:
                if t2_idx in matched_t2_indices:
                    continue
                
                # Ensure t1 and t2 are valid table dicts
                if not isinstance(t1, dict) or not isinstance(t2, dict):
                    logger.warning(f"Skipping invalid table objects for similarity: t1 type {type(t1)}, t2 type {type(t2)}")
                    continue

                score = self._calculate_table_similarity(t1, t2)
                if score > highest_score:
                    highest_score = score
                    best_t2_match_info = (pg2, t2, t2_idx)
            
            if best_t2_match_info:
                pg2_match, t2_match, t2_match_idx = best_t2_match_info
                temp_similarity_matches.append({
                    "t1_info": (pg1, t1, t1_idx),
                    "t2_info": (pg2_match, t2_match, t2_match_idx),
                    "score": highest_score
                })
        
        # Sort by score and resolve conflicts (greedy approach)
        temp_similarity_matches.sort(key=lambda x: x["score"], reverse=True)
        
        matched_t1_indices_sim = set()

        for match_info in temp_similarity_matches:
            pg1, t1, t1_idx = match_info["t1_info"]
            pg2, t2, t2_idx = match_info["t2_info"]

            if t1_idx in matched_t1_indices_sim or t2_idx in matched_t2_indices:
                continue

            status = "moved" if pg1 != pg2 else "modified"
            diff_details = self._compare_individual_tables(t1, t2)
            matches.append({
                "type": "table", "status": status, "score": match_info["score"],
                "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                "page2": pg2, "table2_content": t2.get("content"), "table2_bbox": t2.get("bbox"), "table2_id": t2.get("table_id"),
                "diff_html": diff_details["html"]
            })
            matched_t1_indices_sim.add(t1_idx)
            matched_t2_indices.add(t2_idx)


        # Add tables only in pdf1 (deleted)
        for t1_info in tables1_remaining:
            if t1_info is None: continue # Should have been marked None if matched by hash
            pg1, t1, t1_idx = t1_info
            if t1_idx not in matched_t1_indices_sim:
                diff_details = self._compare_individual_tables(t1, None)
                matches.append({
                    "type": "table", "status": "deleted", "score": 0.0,
                    "page1": pg1, "table1_content": t1.get("content"), "table1_bbox": t1.get("bbox"), "table1_id": t1.get("table_id"),
                    "page2": None, "table2_content": None, "table2_bbox": None, "table2_id": None,
                    "diff_html": diff_details["html"]
                })

        # Add tables only in pdf2 (inserted)
        for pg2, t2, t2_idx in tables2:
            if t2_idx not in matched_t2_indices:
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
        
        # Use content hash for quick check
        if t1.get("content_hash") and t2.get("content_hash") and t1["content_hash"] == t2["content_hash"]:
            return 1.0

        # Cache key using sorted hashes of table content strings
        # This ensures that the order of t1, t2 doesn't affect the cache key
        content1_str = str(t1.get("content"))
        content2_str = str(t2.get("content"))
        hash1 = hashlib.md5(content1_str.encode()).hexdigest()
        hash2 = hashlib.md5(content2_str.encode()).hexdigest()
        cache_key_tuple = tuple(sorted((hash1, hash2)))


        if cache_key_tuple in self.table_comparison_cache:
            return self.table_comparison_cache[cache_key_tuple]

        # Simplified similarity: average of structural and content similarity
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
        
        # Process text matches
        # Keep track of text content that has been part of a "moved" operation to avoid duplication
        moved_text_fingerprints_pdf1 = set() # Hashes of text1 content that moved
        moved_text_fingerprints_pdf2 = set() # Hashes of text2 content that moved

        # First, identify all moved text to populate fingerprints
        for match in text_matches:
            if match["status"] == "moved":
                if match.get("text1"):
                    moved_text_fingerprints_pdf1.add(hashlib.md5(match["text1"].encode()).hexdigest())
                if match.get("text2"):
                     moved_text_fingerprints_pdf2.add(hashlib.md5(match["text2"].encode()).hexdigest())
        
        for match in text_matches:
            status = match["status"]
            text1 = match.get("text1", "")
            text2 = match.get("text2", "")
            page1 = match.get("page1")
            page2 = match.get("page2")
            score = match.get("score", 0.0)

            # Skip deleted/inserted if they are part of an already accounted "moved" operation
            if status == "deleted" and hashlib.md5(text1.encode()).hexdigest() in moved_text_fingerprints_pdf1:
                continue
            if status == "inserted" and hashlib.md5(text2.encode()).hexdigest() in moved_text_fingerprints_pdf2:
                continue

            diff_item = {
                "status": status, "score": score,
                "text1": text1, "text2": text2,
                "page1": page1, "page2": page2,
                "block1_bbox": match.get("block1", {}).get("bbox") if match.get("block1") else None,
                "block2_bbox": match.get("block2", {}).get("bbox") if match.get("block2") else None,
            }

            if status == "moved":
                # For moved items, they affect both source and destination pages in the report
                if page1 is not None:
                    organized_diffs[page1]["text_differences"].append(diff_item)
                if page2 is not None and page1 != page2: # Add to destination page if different
                    # Create a corresponding entry for the destination page
                    # The key is that the *same diff_item* (with both text1 and text2) is used
                    organized_diffs[page2]["text_differences"].append(diff_item)
            elif status == "deleted" and page1 is not None:
                organized_diffs[page1]["text_differences"].append(diff_item)
            elif status == "inserted" and page2 is not None:
                organized_diffs[page2]["text_differences"].append(diff_item)
            elif status == "modified" and page1 is not None: # Modified items appear on their original page
                 organized_diffs[page1]["text_differences"].append(diff_item)
            elif status == "matched" and score < 1.0 and page1 is not None: # Similar items
                 organized_diffs[page1]["text_differences"].append(diff_item)


        # Process table matches
        moved_table_fingerprints_pdf1 = set()
        moved_table_fingerprints_pdf2 = set()

        for match in table_matches:
            if match["status"] == "moved":
                if match.get("table1_content"):
                    moved_table_fingerprints_pdf1.add(hashlib.md5(str(match["table1_content"]).encode()).hexdigest())
                if match.get("table2_content"):
                    moved_table_fingerprints_pdf2.add(hashlib.md5(str(match["table2_content"]).encode()).hexdigest())
        
        for match in table_matches:
            status = match["status"]
            page1 = match.get("page1")
            page2 = match.get("page2")
            table1_content = match.get("table1_content")
            table2_content = match.get("table2_content")
            
            if status == "deleted" and hashlib.md5(str(table1_content).encode()).hexdigest() in moved_table_fingerprints_pdf1:
                continue
            if status == "inserted" and hashlib.md5(str(table2_content).encode()).hexdigest() in moved_table_fingerprints_pdf2:
                continue

            # The diff_html is pre-generated by _compare_individual_tables
            diff_item = {
                "status": status, "score": match.get("score", 0.0),
                "page1": page1, "page2": page2,
                "table1_id": match.get("table1_id"), "table2_id": match.get("table2_id"),
                "table1_bbox": match.get("table1_bbox"), "table2_bbox": match.get("table2_bbox"),
                "diff_html": match.get("diff_html", "")
                # No need to store raw content here if diff_html is comprehensive
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
            # Sort by bounding box y-coordinate primarily, then x-coordinate
            organized_diffs[page_num]["text_differences"].sort(key=lambda d: (
                d.get("block1_bbox")[1] if d.get("block1_bbox") else (d.get("block2_bbox")[1] if d.get("block2_bbox") else float('inf')),
                d.get("block1_bbox")[0] if d.get("block1_bbox") else (d.get("block2_bbox")[0] if d.get("block2_bbox") else float('inf'))
            ))
            organized_diffs[page_num]["table_differences"].sort(key=lambda d: (
                d.get("table1_bbox")[1] if d.get("table1_bbox") else (d.get("table2_bbox")[1] if d.get("table2_bbox") else float('inf')),
                d.get("table1_bbox")[0] if d.get("table1_bbox") else (d.get("table2_bbox")[0] if d.get("table2_bbox") else float('inf'))
            ))
            
        return organized_diffs

