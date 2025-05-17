import os
import re
import hashlib # Added for future use if needed, though not strictly for these changes
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from jinja2 import Environment, BaseLoader, select_autoescape
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReportGenerator:

    SIMILARITY_THRESHOLD_FOR_PAIRING = 0.70 # For pairing text and tables in generate.py

    def __init__(self, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.pdf1_name = ""
        self.pdf2_name = ""

    def _escape(self, text: Optional[Any]) -> str:
        """Basic HTML escaping, ensuring string conversion."""
        if text is None:
            return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _calculate_text_similarity_local(self, text1: Optional[str], text2: Optional[str]) -> float:
        if text1 is None and text2 is None: return 1.0
        if text1 is None or text2 is None: return 0.0
        norm_text1 = text1.strip()
        norm_text2 = text2.strip()
        if not norm_text1 and not norm_text2: return 1.0
        if not norm_text1 or not norm_text2: return 0.0
        return SequenceMatcher(None, text1, text2).ratio()

    def _highlight_word_diff(self, text1: str, text2: str) -> Tuple[str, str]:
        """
        Compares two texts word by word and returns HTML with highlighted differences.
        """
        words1 = text1.split()
        words2 = text2.split()
        matcher = SequenceMatcher(None, words1, words2)

        highlighted_text1_segments = []
        highlighted_text2_segments = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                highlighted_text1_segments.append(self._escape(" ".join(words1[i1:i2])))
                highlighted_text2_segments.append(self._escape(" ".join(words2[j1:j2])))
            elif tag == 'delete':
                highlighted_text1_segments.append(f'<span class="diff-word-deleted">{self._escape(" ".join(words1[i1:i2]))}</span>')
            elif tag == 'insert':
                highlighted_text2_segments.append(f'<span class="diff-word-inserted">{self._escape(" ".join(words2[j1:j2]))}</span>')
            elif tag == 'replace':
                highlighted_text1_segments.append(f'<span class="diff-word-deleted">{self._escape(" ".join(words1[i1:i2]))}</span>')
                highlighted_text2_segments.append(f'<span class="diff-word-inserted">{self._escape(" ".join(words2[j1:j2]))}</span>')

        return " ".join(highlighted_text1_segments), " ".join(highlighted_text2_segments)


    # --- Table Rendering Helpers ---
    def _normalize_cell_for_render(self, cell_text: Any) -> str:
        return str(cell_text).strip().lower()

    def _render_single_html_table_for_pairing(self,
                                             main_content: Optional[List[List[str]]],
                                             compare_content: Optional[List[List[str]]],
                                             is_left_side: bool) -> str:
        if not main_content:
            return "<p>Table not present.</p>"

        html_parts = ['<table class="diff-table" border="1" style="border-collapse: collapse; width: 100%;">']
        num_main_rows = len(main_content)
        num_main_cols = max(len(row) for row in main_content) if main_content and any(main_content) else 0
        num_compare_rows = len(compare_content) if compare_content else 0
        num_compare_cols = max(len(row) for row in compare_content) if compare_content and any(row for row in compare_content) else 0
        max_rows = max(num_main_rows, num_compare_rows)
        max_cols = max(num_main_cols, num_compare_cols)

        if max_cols == 0 and max_rows == 0 :
             html_parts.append("<tr><td>Empty table.</td></tr>")
        else:
            for i in range(max_rows):
                html_parts.append("<tr>")
                for j in range(max_cols):
                    cell_main_text_orig, cell_main_exists = "", False
                    if i < num_main_rows and j < num_main_cols and main_content[i] is not None:
                        cell_main_text_orig, cell_main_exists = str(main_content[i][j]), True

                    cell_compare_text_orig, cell_compare_exists = "", False
                    if compare_content and i < num_compare_rows and j < num_compare_cols and compare_content[i] is not None:
                        cell_compare_text_orig, cell_compare_exists = str(compare_content[i][j]), True

                    cell_display_text = ""
                    cell_class = ""

                    if cell_main_exists:
                        if cell_compare_exists:
                            norm_main = self._normalize_cell_for_render(cell_main_text_orig)
                            norm_compare = self._normalize_cell_for_render(cell_compare_text_orig)
                            if norm_main == norm_compare:
                                cell_class = "cell-similar"
                                cell_display_text = self._escape(cell_main_text_orig)
                            else:
                                cell_class = "cell-modified"
                                if is_left_side:
                                    highlighted_main, _ = self._highlight_word_diff(cell_main_text_orig, cell_compare_text_orig)
                                    cell_display_text = highlighted_main
                                else:
                                    _, highlighted_main_as_compare = self._highlight_word_diff(cell_compare_text_orig, cell_main_text_orig)
                                    cell_display_text = highlighted_main_as_compare
                        else: # Exists in main_content, not in compare_content
                            cell_class = "cell-deleted" if is_left_side else "cell-inserted"
                            cell_display_text = self._escape(cell_main_text_orig)
                    elif cell_compare_exists: # Exists in compare_content, not in main_content
                        cell_class = "cell-inserted" if is_left_side else "cell-deleted"
                        cell_display_text = "&nbsp;" if is_left_side else self._escape(cell_compare_text_orig) # Show content if it's the right side
                        if not is_left_side: # If this is the right side (compare_content is main) and it's an insertion from its perspective
                             cell_display_text = self._escape(cell_compare_text_orig)
                        else: # If this is the left side, it's empty as the corresponding cell was inserted on the right
                            cell_display_text = "&nbsp;"


                    else:
                        cell_display_text = "&nbsp;"
                        cell_class = "cell-empty"

                    html_parts.append(f'<td class="{cell_class}">{cell_display_text}</td>')
                html_parts.append("</tr>")
        html_parts.append("</table>")
        return "".join(html_parts)

    def _generate_paired_table_diff_html(self,
                                         table1_content: Optional[List[List[str]]],
                                         table2_content: Optional[List[List[str]]],
                                         table1_id: Optional[str],
                                         table2_id: Optional[str]) -> str:
        html = ['<div class="table-diff-container" style="display: flex; width: 100%;">']
        # Left side (PDF1)
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        pdf1_table_name = table1_id if table1_id else "Table from PDF 1"
        html.append(f'<h6>{self._escape(pdf1_table_name)} ({self.pdf1_name})</h6>')
        html.append(self._render_single_html_table_for_pairing(table1_content, table2_content, is_left_side=True))
        html.append('</div>')

        # Separator
        html.append('<div style="width: 1px; background-color: #ccc;"></div>')

        # Right side (PDF2)
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        pdf2_table_name = table2_id if table2_id else "Table from PDF 2"
        html.append(f'<h6>{self._escape(pdf2_table_name)} ({self.pdf2_name})</h6>')
        html.append(self._render_single_html_table_for_pairing(table2_content, table1_content, is_left_side=False))
        html.append('</div>')

        html.append('</div>')
        return "".join(html)

    def _has_true_differences(self, diff_list: List[Dict]) -> bool:
        """Checks if there are differences other than 100% score moved items."""
        if not diff_list:
            return False
        for diff in diff_list:
            # If it's a 'moved' item with a score of 1.0 (100% similarity), ignore it for "Has Differences" badge
            if diff.get("status") == "moved" and diff.get("score", 0.0) == 1.0:
                continue
            # Any other type of difference (modified, inserted, deleted, or moved < 100%, or matched < 100%) counts
            if diff.get("status") != "matched" or (diff.get("status") == "matched" and diff.get("score", 0.0) < 1.0) :
                 return True
        return False # Only 100% moved items or no items or only matched items with 100%

    def generate_html_report(
        self,
        results: Dict,
        pdf1_name: str,
        pdf2_name: str,
        metadata: Optional[Dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> str:
        self.pdf1_name = pdf1_name
        self.pdf2_name = pdf2_name
        if progress_callback: progress_callback(0.1)

        processed_pages = {}
        total_pages_to_process = len(results.get("pages", {}))
        pages_processed_count = 0

        # Ensure pages are sorted numerically for consistent processing and nav generation
        sorted_page_keys = sorted(results.get("pages", {}).keys(), key=lambda x: int(str(x).split('_')[-1]))


        for pg_num_key in sorted_page_keys:
            page_data = results.get("pages", {})[pg_num_key]
            # Assuming pg_num_key can be directly used or converted to the display page number
            # If pg_num_key is like "page_1", extract "1"
            pg_num_display = str(pg_num_key).split('_')[-1]


            text_diffs_for_render = self._prepare_text_diffs_for_render(page_data.get("text_differences", []), pg_num_display)
            table_diffs_for_render = self._prepare_table_diffs_for_render(page_data.get("table_differences", []), pg_num_display)

            page_has_actual_text_differences = self._has_true_differences(text_diffs_for_render)
            page_has_actual_table_differences = self._has_true_differences(table_diffs_for_render)

            page_has_actual_differences = page_has_actual_text_differences or page_has_actual_table_differences

            processed_pages[pg_num_display] = { # Use display number as key for template
                "text_differences": text_diffs_for_render,
                "table_differences": table_diffs_for_render,
                "has_actual_differences": page_has_actual_differences
            }
            pages_processed_count += 1
            if progress_callback and total_pages_to_process > 0:
                progress_callback(0.1 + 0.8 * (pages_processed_count / total_pages_to_process))

        summary_stats = self._calculate_summary(results)

        env = Environment(loader=BaseLoader(), autoescape=select_autoescape(['html', 'xml']))
        env.filters['escapejs'] = lambda s: self._escape(s).replace("'", "\\'")
        tpl = env.from_string(self._TEMPLATE)

        # Ensure pages are passed to template sorted by the display page number for nav consistency
        sorted_processed_pages_for_template = {
            k: processed_pages[k] for k in sorted(processed_pages.keys(), key=lambda x: int(x))
        }

        html_output = tpl.render(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pdf1_name=self.pdf1_name, pdf2_name=self.pdf2_name,
            pages=sorted_processed_pages_for_template, # Pass sorted pages
            meta=metadata or {}, summary=summary_stats,
            escape_html=self._escape
        )
        if progress_callback: progress_callback(0.95)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_pdf1_name = re.sub(r'[^\w\-_.]', '_', pdf1_name)
        safe_pdf2_name = re.sub(r'[^\w\-_.]', '_', pdf2_name)
        outfile_name = f"{safe_pdf1_name}_vs_{safe_pdf2_name}_{ts}.html"
        fullpath = os.path.join(self.output_dir, outfile_name)
        with open(fullpath, "w", encoding="utf-8") as fh: fh.write(html_output)
        logger.info(f"Report generated: {fullpath}")
        if progress_callback: progress_callback(1.0)
        return fullpath

    def _prepare_text_diffs_for_render(self, diffs: List[Dict], current_page_num_str: str) -> List[Dict]:
        current_page_num = int(current_page_num_str) # Convert to int for comparison
        pre_classified_items = [d for d in diffs if d["status"] not in ["deleted", "inserted"]]
        deleted_items = [d for d in diffs if d["status"] == "deleted"]
        inserted_items = [d for d in diffs if d["status"] == "inserted"]
        candidate_diffs_for_page = list(pre_classified_items)
        consumed_deleted_indices, consumed_inserted_indices = set(), set()

        for del_idx, d_item in enumerate(deleted_items):
            if del_idx in consumed_deleted_indices: continue
            d_text1, d_page1 = d_item.get("text1", ""), d_item.get("page1")
            d_page1_int = int(d_page1) if d_page1 is not None else None

            best_i_match, best_i_idx, highest_sim = None, -1, -1.0
            for ins_idx, i_item in enumerate(inserted_items):
                if ins_idx in consumed_inserted_indices: continue
                i_text2 = i_item.get("text2", "")
                similarity = self._calculate_text_similarity_local(d_text1, i_text2)
                if similarity > highest_sim: # Prioritize higher similarity
                    highest_sim, best_i_match, best_i_idx = similarity, i_item, ins_idx
            # If a good match is found, pair them as modified or moved
            if best_i_match and highest_sim >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                consumed_deleted_indices.add(del_idx)
                consumed_inserted_indices.add(best_i_idx)
                i_page2 = best_i_match.get("page2")
                i_page2_int = int(i_page2) if i_page2 is not None else None
                paired_item = {
                    "text1": d_text1, "text2": best_i_match.get("text2", ""), "score": highest_sim,
                    "page1": d_page1, "page2": i_page2, # Keep original string page numbers for display
                    "block1_bbox": d_item.get("block1_bbox"), "block2_bbox": best_i_match.get("block2_bbox"),
                    "status": "modified" if d_page1_int == i_page2_int else "moved"
                }
                candidate_diffs_for_page.append(paired_item)

        # Add remaining deleted/inserted items that weren't paired
        for del_idx, d_item in enumerate(deleted_items):
            if del_idx not in consumed_deleted_indices: candidate_diffs_for_page.append(d_item)
        for ins_idx, i_item in enumerate(inserted_items):
            if ins_idx not in consumed_inserted_indices: candidate_diffs_for_page.append(i_item)

        # Sort all items for display on the current page
        def get_sort_key(item):
            b1, b2 = item.get("block1_bbox"), item.get("block2_bbox")
            y, x = float('inf'), float('inf')
            page1_val = item.get("page1")
            page2_val = item.get("page2")
            page1_int = int(page1_val) if page1_val is not None else float('inf')
            page2_int = int(page2_val) if page2_val is not None else float('inf')

            if current_page_num == page1_int and b1: y,x = b1[1],b1[0]
            elif current_page_num == page2_int and b2: y,x = b2[1],b2[0]
            elif b1: y,x = b1[1],b1[0]
            elif b2: y,x = b2[1],b2[0]
            return (y,x)
        candidate_diffs_for_page.sort(key=get_sort_key)

        final_rendered_list = []
        for item in candidate_diffs_for_page:
            status = item["status"]
            text1, text2 = item.get("text1", ""), item.get("text2", "")
            page1, page2 = item.get("page1"), item.get("page2") # String page numbers
            page1_int = int(page1) if page1 is not None else None
            page2_int = int(page2) if page2 is not None else None
            score = item.get("score", 0.0)
            display_item, relevant_for_this_page = item.copy(), False

            if status == "modified":
                if current_page_num == page1_int or (page1_int is None and current_page_num == page2_int):
                    relevant_for_this_page = True
                    highlighted1, highlighted2 = self._highlight_word_diff(text1, text2)
                    display_item["display_text1"] = highlighted1
                    display_item["display_text2"] = highlighted2
            elif status == "moved":
                if current_page_num == page1_int or current_page_num == page2_int:
                    relevant_for_this_page = True
                    if score < 1.0 and score >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                        h1, h2 = self._highlight_word_diff(text1, text2)
                        display_item["display_text1"] = h1
                        display_item["display_text2"] = h2
                    else:
                        display_item["display_text1"] = self._escape(text1)
                        display_item["display_text2"] = self._escape(text2)
                    display_item["move_info"] = f"Moved: {self.pdf1_name} p. {page1} ➜ {self.pdf2_name} p. {page2} (Similarity: {score*100:.2f}%)"

            elif status == "deleted":
                if current_page_num == page1_int:
                    relevant_for_this_page = True
                    display_item["display_text1"] = self._escape(text1)
                    display_item["display_text2"] = f"<i>Not present in {self.pdf2_name}</i>"
            elif status == "inserted":
                if current_page_num == page2_int:
                    relevant_for_this_page = True
                    display_item["display_text1"] = f"<i>Not present in {self.pdf1_name}</i>"
                    display_item["display_text2"] = self._escape(text2)
            elif status == "matched":
                if score < 1.0:
                    if current_page_num == page1_int or (page1_int is None and current_page_num == page2_int):
                        relevant_for_this_page = True
                        highlighted1, highlighted2 = self._highlight_word_diff(text1, text2)
                        display_item["display_text1"] = highlighted1
                        display_item["display_text2"] = highlighted2

            if relevant_for_this_page: final_rendered_list.append(display_item)
        return final_rendered_list

    def _prepare_table_diffs_for_render(self, diffs: List[Dict], current_page_num_str: str) -> List[Dict]:
        current_page_num = int(current_page_num_str) # Convert to int
        pre_classified_items = [d for d in diffs if d["status"] not in ["deleted", "inserted"]]
        deleted_tables = [d for d in diffs if d["status"] == "deleted"]
        inserted_tables = [d for d in diffs if d["status"] == "inserted"]
        candidate_table_diffs = list(pre_classified_items)
        consumed_deleted_table_indices, consumed_inserted_table_indices = set(), set()

        for del_idx, d_table in enumerate(deleted_tables):
            if del_idx in consumed_deleted_table_indices: continue
            d_content1, d_page1, d_id1 = d_table.get("table1_content"), d_table.get("page1"), d_table.get("table1_id")
            d_page1_int = int(d_page1) if d_page1 is not None else None
            best_i_match, best_i_idx, highest_sim = None, -1, -1.0

            for ins_idx, i_table in enumerate(inserted_tables):
                if ins_idx in consumed_inserted_table_indices: continue
                i_content2 = i_table.get("table2_content")
                sim_score = 0.0
                if d_content1 and i_content2:
                    flat1 = " ".join([" ".join(map(str, r)) for r in d_content1 if r])
                    flat2 = " ".join([" ".join(map(str, r)) for r in i_content2 if r])
                    sim_score = self._calculate_text_similarity_local(flat1, flat2)

                if sim_score > highest_sim:
                    highest_sim, best_i_match, best_i_idx = sim_score, i_table, ins_idx

            if best_i_match and highest_sim >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                consumed_deleted_table_indices.add(del_idx)
                consumed_inserted_table_indices.add(best_i_idx)
                i_page2, i_content2, i_id2 = best_i_match.get("page2"), best_i_match.get("table2_content"), best_i_match.get("table2_id")
                i_page2_int = int(i_page2) if i_page2 is not None else None
                new_status = "modified" if d_page1_int == i_page2_int else "moved"
                paired_html = self._generate_paired_table_diff_html(d_content1, i_content2, d_id1, i_id2)
                candidate_table_diffs.append({
                    "status": new_status, "score": highest_sim, "page1": d_page1, "page2": i_page2,
                    "table1_id": d_id1, "table2_id": i_id2, "table1_bbox": d_table.get("table1_bbox"),
                    "table2_bbox": best_i_match.get("table2_bbox"), "diff_html": paired_html,
                    "table1_content": d_content1, "table2_content": i_content2,
                })

        for del_idx, d_table in enumerate(deleted_tables):
            if del_idx not in consumed_deleted_table_indices:
                d_table["diff_html"] = self._generate_paired_table_diff_html(d_table.get("table1_content"), None, d_table.get("table1_id"), None)
                candidate_table_diffs.append(d_table)
        for ins_idx, i_table in enumerate(inserted_tables):
            if ins_idx not in consumed_inserted_table_indices:
                i_table["diff_html"] = self._generate_paired_table_diff_html(None, i_table.get("table2_content"), None, i_table.get("table2_id"))
                candidate_table_diffs.append(i_table)

        def get_table_sort_key(item):
            b1,b2=item.get("table1_bbox"),item.get("table2_bbox")
            y,x=float('inf'),float('inf')
            page1_val = item.get("page1")
            page2_val = item.get("page2")
            page1_int = int(page1_val) if page1_val is not None else float('inf')
            page2_int = int(page2_val) if page2_val is not None else float('inf')

            if current_page_num==page1_int and b1:y,x=b1[1],b1[0]
            elif current_page_num==page2_int and b2:y,x=b2[1],b2[0]
            elif b1:y,x=b1[1],b1[0]
            elif b2:y,x=b2[1],b2[0]
            return(y,x)
        candidate_table_diffs.sort(key=get_table_sort_key)

        final_rendered_list = []
        for item in candidate_table_diffs:
            status,p1_str,p2_str=item["status"],item.get("page1"),item.get("page2")
            p1_int = int(p1_str) if p1_str is not None else None
            p2_int = int(p2_str) if p2_str is not None else None
            disp_item,relevant_for_this_page = item.copy(),False

            if status == "moved":
                if current_page_num == p1_int or current_page_num == p2_int:
                    relevant_for_this_page = True
                    disp_item["move_info"] = f"Moved: {self.pdf1_name} p. {p1_str} ➜ {self.pdf2_name} p. {p2_str} (Similarity: {item.get('score',0)*100:.2f}%)"
                    if not disp_item.get("diff_html") and item.get("table1_content") and item.get("table2_content"):
                        disp_item["diff_html"] = self._generate_paired_table_diff_html(
                            item.get("table1_content"), item.get("table2_content"),
                            item.get("table1_id"), item.get("table2_id")
                        )
            elif status == "modified":
                 if current_page_num == p1_int or (p1_int is None and current_page_num == p2_int):
                    relevant_for_this_page = True
                    if not disp_item.get("diff_html") and item.get("table1_content") and item.get("table2_content"):
                         disp_item["diff_html"] = self._generate_paired_table_diff_html(
                            item.get("table1_content"), item.get("table2_content"),
                            item.get("table1_id"), item.get("table2_id")
                        )
            elif status == "deleted":
                 if current_page_num == p1_int:
                    relevant_for_this_page = True
                    if not disp_item.get("diff_html"):
                         disp_item["diff_html"] = self._generate_paired_table_diff_html(item.get("table1_content"), None, item.get("table1_id"), None)
            elif status == "inserted":
                 if current_page_num == p2_int:
                    relevant_for_this_page = True
                    if not disp_item.get("diff_html"):
                         disp_item["diff_html"] = self._generate_paired_table_diff_html(None, item.get("table2_content"), None, item.get("table2_id"))
            elif status == "matched":
                if item.get("score", 1.0) < 1.0:
                    if current_page_num == p1_int or (p1_int is None and current_page_num == p2_int):
                        relevant_for_this_page = True
                        if not disp_item.get("diff_html") and item.get("table1_content") and item.get("table2_content"):
                            disp_item["diff_html"] = self._generate_paired_table_diff_html(
                                item.get("table1_content"), item.get("table2_content"),
                                item.get("table1_id"), item.get("table2_id")
                            )
            if relevant_for_this_page:
                final_rendered_list.append(disp_item)
        return final_rendered_list


    def _calculate_summary(self, results: Dict) -> Dict:
        total_pages_in_docs = results.get("max_pages", 0)
        pages_with_reportable_diff = 0
        unique_text_diff_keys, unique_table_diff_keys = set(), set()

        for pg_num_key, page_data in results.get("pages", {}).items():
            current_page_has_reportable_diff = False

            for diff in page_data.get("text_differences", []):
                status = diff.get("status")
                score = diff.get("score", 1.0)
                is_reportable_summary_diff = False
                if status == "moved":
                    is_reportable_summary_diff = True
                elif status != "matched" or (status == "matched" and score < 1.0):
                    is_reportable_summary_diff = True

                if is_reportable_summary_diff:
                    current_page_has_reportable_diff = True
                    b1_val, b2_val = diff.get("block1"), diff.get("block2")
                    p1_val, p2_val = diff.get("page1"), diff.get("page2")
                    key_parts = [
                        str(status if status is not None else "s_unknown"),
                        str(b1_val.get("hash") if isinstance(b1_val, dict) and b1_val.get("hash") is not None else hashlib.md5(str(b1_val).encode()).hexdigest() if b1_val else "h1_none"),
                        str(b2_val.get("hash") if isinstance(b2_val, dict) and b2_val.get("hash") is not None else hashlib.md5(str(b2_val).encode()).hexdigest() if b2_val else "h2_none"),
                        str(p1_val if p1_val is not None else "p1_none"),
                        str(p2_val if p2_val is not None else "p2_none")
                    ]
                    unique_text_diff_keys.add(tuple(sorted(key_parts)))

            for diff in page_data.get("table_differences", []):
                status = diff.get("status")
                score = diff.get("score", 1.0)
                is_reportable_summary_diff = False
                if status == "moved":
                    is_reportable_summary_diff = True
                elif status != "matched" or (status == "matched" and score < 1.0):
                    is_reportable_summary_diff = True

                if is_reportable_summary_diff:
                    current_page_has_reportable_diff = True
                    t1id_val, t2id_val = diff.get("table1_id"), diff.get("table2_id")
                    p1_val, p2_val = diff.get("page1"), diff.get("page2")
                    key_parts = [
                        str(status if status is not None else "s_unknown_tbl"),
                        str(t1id_val if t1id_val is not None else "id1_none_tbl"),
                        str(t2id_val if t2id_val is not None else "id2_none_tbl"),
                        str(p1_val if p1_val is not None else "p1_none_tbl"),
                        str(p2_val if p2_val is not None else "p2_none_tbl")
                    ]
                    unique_table_diff_keys.add(tuple(sorted(key_parts)))

            if current_page_has_reportable_diff:
                pages_with_reportable_diff +=1

        return {
            "total_pages": total_pages_in_docs,
            "pages_with_differences": pages_with_reportable_diff,
            "text_differences": len(unique_text_diff_keys),
            "table_differences": len(unique_table_diff_keys),
        }

    _TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDF Comparison: {{ pdf1_name }} vs {{ pdf2_name }}</title>
<style>
body{font-family:'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;background:#f4f6f8;margin:0;padding:20px;color:#3a3b3c;line-height:1.6}
.container{max-width:1200px;margin:20px auto;background:#ffffff;padding:25px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,.07)}
header{background:#C62828;color:#fff;padding:25px 30px;margin:-25px -25px 25px -25px;border-radius:10px 10px 0 0}
header h1 {font-size: 2em; font-weight: 600; margin-bottom: 8px; color: #fff; text-shadow: 1px 1px 3px rgba(0,0,0,0.3);}
header p {font-size: 1em; opacity: 0.95; margin:0; color: #f1f1f1; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }

/* Dashboard-like Summary Section */
.summary{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
    gap: 25px;
    margin: 30px 0;
    padding: 0;
}
.stat-item{
    background: #ffffff;
    border-radius: 8px;
    padding: 25px;
    text-align: left;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    transition: transform 0.25s ease-in-out, box-shadow 0.25s ease-in-out;
    position: relative;
}
.stat-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 7px 20px rgba(0,0,0,0.1);
}

/* Specific colors for stat items */
.stat-item.total-pages { border-left: 6px solid #4CAF50; } /* Green */
.stat-item.pages-diff { border-left: 6px solid #F44336; } /* Red */
.stat-item.text-diff { border-left: 6px solid #FFC107; }  /* Amber/Yellow */
.stat-item.table-diff { border-left: 6px solid #FF9800; } /* Orange */

.stat-label{
    font-size: 0.95em;
    color: #6c757d;
    margin-bottom: 10px;
    text-transform: uppercase;
    font-weight: 600;
}
.stat-value{
    font-size: 2.5em;
    font-weight: 700;
    line-height: 1.1;
    display: block;
}

/* Matching stat value colors to their accent borders */
.stat-item.total-pages .stat-value { color: #4CAF50; }
.stat-item.pages-diff .stat-value { color: #F44336; }
.stat-item.text-diff .stat-value { color: #FFC107; }
.stat-item.table-diff .stat-value { color: #FF9800; }

/* End Dashboard-like Summary Section */

.diff-legend{display:flex;flex-wrap:wrap;gap:12px;padding:15px;background:#f8f9fa;border-radius:6px;margin:25px 0; border: 1px solid #e9ecef;}
.legend-item{display:flex;align-items:center;font-size:.9em}
.legend-color{width:18px;height:18px;border-radius:4px;margin-right:8px; border: 1px solid rgba(0,0,0,0.1);}
.page-nav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:25px;position:sticky;top:0;background:rgba(255,255,255,0.95);padding:12px 0;z-index:100;border-bottom:1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}

.page-button{
    padding:9px 14px;
    background:#f8f9fa;
    border:1px solid #dee2e6;
    border-bottom:none;
    border-radius:5px 5px 0 0;
    text-decoration:none;
    color:#495057;
    font-weight:500;
    transition: background-color 0.2s ease, color 0.2s ease, border-color 0.2s ease;
}
.page-button:hover{
    background:#e9ecef;
    color:#212529;
    border-color: #ced4da;
}
.page-button.active{
    background:#D32F2F;
    color:#fff;
    border-color:#B71C1C;
    font-weight:600;
}

.page-section{margin-bottom:25px;border:1px solid #e9ecef;border-radius:8px;padding:0; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
.page-header{background:#f8f9fa;padding:14px 20px;margin:0;border-bottom:1px solid #e9ecef;display:flex;justify-content:space-between;align-items:center; border-radius:8px 8px 0 0;}
.page-title{color:#B71C1C; font-size:1.4em; font-weight:600;}
.page-section > h3 { padding: 12px 20px; margin:0; font-size: 1.15em; background-color: #fcfdfe; color: #343a40; border-bottom: 1px solid #f1f3f5;}
.difference-item{margin: 20px;border:1px solid #f1f3f5;border-radius:6px;overflow:hidden; background: #fff;}
.difference-header{padding:12px 18px;background:#fbfcfd;border-bottom:1px solid #f1f3f5;font-weight:600;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap; gap: 10px; font-size: 0.95em;}
.difference-content{display:flex;width:100%}
.difference-side{flex:1;padding:15px 18px;border-right:1px solid #f1f3f5;overflow-x:auto; min-width: 45%;}
.difference-side:last-child{border-right:none}
.side-header{font-weight:600;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #f1f3f5; font-size:1em; color: #495057;}
.status-badge{display:inline-block;padding:5px 10px;border-radius:15px;font-size:11px;font-weight:600;text-transform:uppercase; line-height:1.2;}

.status-deleted{background:#FFEBEE;color:#C62828} /* Used for "Has Differences" */
.status-modified{background:#FFF9C4;color:#F57F17} /* Light Yellow */
.status-inserted{background:#E0F2F1;color:#00796B} /* Teal */
.status-moved{background:#E3F2FD;color:#1E88E5}   /* Light Blue */
.status-matched{background:#F5F5F5;color:#555555} /* Neutral Grey */


.move-info{font-style:italic;color:#5f6368;margin-left:auto; font-weight:normal; font-size:0.9em;}
.move-info em, .move-info i {font-style:normal;font-weight:bold;color:#B71C1C}

.text-deleted, .text-inserted, .text-modified, .text-matched { padding: 6px 8px; white-space: pre-wrap; word-wrap: break-word; border-radius:4px; font-size:0.9em; margin-top:3px; margin-bottom:3px; display:block;}
.text-deleted { background:#FFEBEE; color:#C62828; border-left: 3px solid #C62828;}
.text-inserted { background:#FFFDE7; color:#E65100; border-left: 3px solid #F9A825;} /* Orange for inserted text */
.text-modified { border-left: 3px solid #FFC107; background: #FFF9E6; } /* Amber/Yellow for modified text blocks */
.text-matched { background:#F8F9FA; color:#495057; border-left: 3px solid #adb5bd;} 

.diff-word-deleted { background-color: #ef9a9a; color: #7f0000; font-weight: bold; padding: 1px 3px; border-radius:3px;}
.diff-word-inserted { background-color: #fff59d; color: #bf360c; font-weight: bold; padding: 1px 3px; border-radius:3px;}


.table-diff-container{display:flex;width:100%;}
.table-diff-side{flex:1;padding:12px;overflow-x:auto;}
.table-diff-side h6 { margin-top:0; margin-bottom:12px; font-size: 1em; color: #343a40; }
.diff-table{width:100%;border-collapse:collapse;font-size:0.85em;table-layout:auto;} /* Smaller font for tables */
.diff-table th,.diff-table td{border:1px solid #e9ecef;padding:10px;text-align:left;word-wrap:break-word;}
.diff-table th{background:#f8f9fa;font-weight:600; color: #495057;}
.cell-similar{background:#F1F8E9!important;border:1px solid #DCEDC8!important} 
.cell-modified { background:#FFF9E6!important; border:1px solid #FFECB3!important; } /* Amber/Yellow */
.cell-deleted{background:#FFEBEE!important;border:1px solid #FFCDD2!important} /* Red */
.cell-inserted{background:#FFFDE7!important;border:1px solid #FFE0B2!important} /* Orange */
.cell-empty { background: #ffffff; }

.metadata-table{width:100%;border-collapse:collapse;margin-bottom:25px; border-radius: 8px; overflow:hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05);}
.metadata-table th,.metadata-table td{border:1px solid #e9ecef;padding:12px 15px;text-align:left}
.metadata-table th{background:#f8f9fa;width:220px; font-weight:600; color: #495057;}
@media(max-width:768px){
.difference-content,.table-diff-container{flex-direction:column}
.difference-side,.table-diff-side{border-right:none;border-bottom:1px solid #eee;margin-bottom:10px; min-width: 100%;}
.difference-side:last-child,.table-diff-side:last-child{border-bottom:none;margin-bottom:0}
.summary {grid-template-columns: 1fr;} /* Stack summary items on small screens */
header h1 {font-size: 1.6em;}
.page-title {font-size: 1.2em;}
.stat-value {font-size: 2em;}
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function () {
    const pageButtons = document.querySelectorAll('.page-button');
    const pageSections = document.querySelectorAll('.page-section');
    let currentActiveButton = null;

    function setActiveButton(buttonToActivate) {
        if (currentActiveButton) {
            currentActiveButton.classList.remove('active');
        }
        if (buttonToActivate) {
            buttonToActivate.classList.add('active');
            currentActiveButton = buttonToActivate;
        } else {
            currentActiveButton = null;
        }
    }

    function updateActiveButtonBasedOnHash(hash) {
        const targetButton = document.querySelector(`.page-button[href="${hash}"]`);
        setActiveButton(targetButton);
    }

    // Initial active state based on hash or first button
    if (window.location.hash) {
        updateActiveButtonBasedOnHash(window.location.hash);
    } else if (pageButtons.length > 0) {
        setActiveButton(pageButtons[0]); // Activate the first page button by default
         // Optionally, update hash to reflect this default active page
        if (history.replaceState) {
            history.replaceState(null, null, pageButtons[0].getAttribute('href'));
        }
    }

    // Listen for hash changes (e.g., clicking a page button)
    window.addEventListener('hashchange', function () {
        updateActiveButtonBasedOnHash(window.location.hash);
    });

    // Intersection Observer to update active button on scroll
    const observerOptions = {
        root: null, // relative to document viewport
        rootMargin: '0px 0px -75% 0px', // Trigger when 25% of the section is visible from the top
        threshold: 0.01 // A small part of the section needs to be visible
    };

    const observerCallback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const sectionId = entry.target.id;
                const correspondingButton = document.querySelector(`.page-button[href="#${sectionId}"]`);
                setActiveButton(correspondingButton);
                // Update URL hash without adding to history, only if not already matching
                // This helps if user scrolls to a section without clicking a nav link
                if (window.location.hash !== `#${sectionId}`) {
                     if (history.replaceState) { // Check if replaceState is supported
                        // history.replaceState(null, null, `#${sectionId}`); // Commented out to prevent scroll jump on some browsers
                     }
                }
            }
        });
    };

    const observer = new IntersectionObserver(observerCallback, observerOptions);
    pageSections.forEach(section => {
        observer.observe(section);
    });

    // Smooth scroll for page navigation links
    pageButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault(); // Prevent default anchor click behavior
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            if (targetSection) {
                // Calculate position of target section, considering the sticky header height
                const headerOffset = document.querySelector('.page-nav').offsetHeight + 15; // Increased offset slightly
                const elementPosition = targetSection.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth"
                });
                 // Manually update hash as smooth scroll might not trigger hashchange immediately or consistently
                if (history.pushState) { // Use pushState to allow back button navigation
                    history.pushState(null, null, targetId);
                } else {
                    window.location.hash = targetId; // Fallback for older browsers
                }
                updateActiveButtonBasedOnHash(targetId); // Ensure button is active immediately
            }
        });
    });
});
</script>
</head>
<body>
<div class="container">
<header>
  <h1>PDF Comparison Report</h1> <p>Comparing: <strong>{{ pdf1_name }}</strong> vs <strong>{{ pdf2_name }}</strong></p>
  <p style="font-size:0.8em; opacity: 0.8;">Generated on: {{ now }}</p>
</header>

<table class="metadata-table">
  <tr><th>Document 1</th><td>{{ pdf1_name }}</td></tr>
  <tr><th>Document 2</th><td>{{ pdf2_name }}</td></tr>
  {% for k,v in meta.items() %}
  <tr><th>{{ k }}</th><td>{{ escape_html(v) }}</td></tr>
  {% endfor %}
</table>

<div class="summary">
  <div class="stat-item total-pages">
    <div class="stat-label">Total Pages</div>
    <div class="stat-value">{{ summary.total_pages }}</div>
  </div>
  <div class="stat-item pages-diff">
    <div class="stat-label">Pages with Differences</div>
    <div class="stat-value">{{ summary.pages_with_differences }}</div>
  </div>
  <div class="stat-item text-diff">
    <div class="stat-label">Text Differences</div>
    <div class="stat-value">{{ summary.text_differences }}</div>
  </div>
  <div class="stat-item table-diff">
    <div class="stat-label">Table Differences</div>
    <div class="stat-value">{{ summary.table_differences }}</div>
  </div>
</div>

<div class="diff-legend">
  <div class="legend-item"><div class="legend-color" style="background:#FFEBEE;border:1px solid #EF9A9A"></div><span>Deleted (PDF1 only)</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#FFFDE7;border:1px solid #FFF59D"></div><span>Inserted (PDF2 only)</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#FFF3E0;border:1px solid #FFCC80"></div><span>Modified</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#F5F5F5;border:1px solid #BDBDBD"></div><span>Similar / Matched</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#E3F2FD;border:1px solid #90CAF9"></div><span>Moved</span></div>
</div>

<div class="page-nav">
  {# Ensure pages are iterated in numerical order for navigation buttons #}
  {% for pg_num in pages.keys()|map('int')|sort %}
    <a href="#page-{{ pg_num }}" class="page-button">Page {{ pg_num }}</a>
  {% endfor %}
</div>

{# Ensure page content is displayed in numerical order #}
{% for pg_num_str, page_content in pages.items()|sort(attribute='0') %}
{% set pg_num = pg_num_str|int %} {# Convert string page number to int for ID consistency #}
<section id="page-{{ pg_num }}" class="page-section">
  <div class="page-header">
    <h2 class="page-title">Page {{ pg_num }} Details</h2>
    {% if page_content.has_actual_differences %}
      <span class="status-badge status-deleted">Has Differences</span>
    {% else %}
      <span class="status-badge status-matched">No Notable Differences</span>
    {% endif %}
  </div>

  {% if page_content.text_differences %}
    <h3>Text Differences ({{ page_content.text_differences|length }})</h3>
    {% for d in page_content.text_differences %}
    <div class="difference-item">
      <div class="difference-header">
        <span class="status-badge status-{{ d.status }}">{{ d.status }}</span>
        {% if d.score is defined and (d.status == 'modified' or d.status == 'matched' or (d.status == 'moved' and d.score < 1.0)) %}
          <span>Similarity: {{ "%.2f"|format(d.score*100) }}%</span>
        {% endif %}
        {% if d.move_info %}<span class="move-info">{{ d.move_info|safe }}</span>{% endif %}
      </div>
      <div class="difference-content">
        <div class="difference-side">
          <div class="side-header">{{ pdf1_name }} (Page {{ d.page1 if d.page1 else 'N/A' }})</div>
          <div class="text-{{ d.status }}">{{ d.display_text1|safe }}</div>
        </div>
        <div class="difference-side">
          <div class="side-header">{{ pdf2_name }} (Page {{ d.page2 if d.page2 else 'N/A' }})</div>
          <div class="text-{{ d.status }}">{{ d.display_text2|safe }}</div>
        </div>
      </div>
    </div>
    {% endfor %}
  {% endif %}

  {% if page_content.table_differences %}
    <h3>Table Differences ({{ page_content.table_differences|length }})</h3>
    {% for d in page_content.table_differences %}
    <div class="difference-item">
      <div class="difference-header">
        <span class="status-badge status-{{ d.status }}">{{ d.status }}</span>
        {% if d.score is defined and (d.status == 'modified' or d.status == 'matched' or (d.status == 'moved' and d.score < 1.0)) %}
          <span>Similarity: {{ "%.2f"|format(d.score*100) }}%</span>
        {% endif %}
        {% if d.move_info %}<span class="move-info">{{ d.move_info|safe }}</span>{% endif %}
      </div>
      <div class="difference-content">
        {{ d.diff_html|safe }}
      </div>
    </div>
    {% endfor %}
  {% endif %}

  {% if not page_content.text_differences and not page_content.table_differences %}
    <p style="padding:15px 20px; text-align:center; color:#757575; font-size: 0.95em;">No differences found on this page.</p>
  {% endif %}
</section>
{% endfor %}
</div>
</body>
</html>
"""
