import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from jinja2 import Environment, BaseLoader, select_autoescape
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ReportGenerator:

    SIMILARITY_THRESHOLD_FOR_PAIRING = 0.70

    def __init__(self, output_dir: str = "reports"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.pdf1_name = ""
        self.pdf2_name = ""

    def _escape(self, text: Optional[Any]) -> str:
        if text is None:
            return ""
        return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _calculate_text_similarity_local(self, text1: Optional[str], text2: Optional[str]) -> float:
        if text1 is None and text2 is None: return 1.0
        if text1 is None or text2 is None: return 0.0
        norm_text1 = str(text1).strip()
        norm_text2 = str(text2).strip()
        if not norm_text1 and not norm_text2: return 1.0
        if not norm_text1 or not norm_text2: return 0.0
        return SequenceMatcher(None, norm_text1, norm_text2).ratio()

    def _highlight_word_diff(self, text1: str, text2: str) -> Tuple[str, str]:
        words1 = str(text1).split()
        words2 = str(text2).split()
        matcher = SequenceMatcher(None, words1, words2)
        
        highlighted_text1_segments = []
        highlighted_text2_segments = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            seg1 = " ".join(words1[i1:i2])
            seg2 = " ".join(words2[j1:j2])
            if tag == 'equal':
                highlighted_text1_segments.append(self._escape(seg1))
                highlighted_text2_segments.append(self._escape(seg2))
            elif tag == 'delete':
                highlighted_text1_segments.append(f'<span class="diff-word-deleted">{self._escape(seg1)}</span>')
            elif tag == 'insert':
                highlighted_text2_segments.append(f'<span class="diff-word-inserted">{self._escape(seg2)}</span>')
            elif tag == 'replace':
                highlighted_text1_segments.append(f'<span class="diff-word-deleted">{self._escape(seg1)}</span>')
                highlighted_text2_segments.append(f'<span class="diff-word-inserted">{self._escape(seg2)}</span>')
        
        return " ".join(highlighted_text1_segments), " ".join(highlighted_text2_segments)

    def _normalize_cell_for_render(self, cell_text: Any) -> str:
        return str(cell_text).strip().lower()

    def _render_single_html_table_for_pairing(self, 
                                             main_content: Optional[List[List[str]]],
                                             compare_content: Optional[List[List[str]]],
                                             is_left_side: bool) -> str:
        if main_content is None: # Check specifically for None if the entire table for this side is absent
            return "<p>Table data not available for this side.</p>"
        if not main_content: # Check for empty list (e.g. table exists but has no rows)
             return "<p>Table is present but empty.</p>"

        html_parts = ['<table class="diff-table" border="1">']
        num_main_rows = len(main_content)
        num_main_cols = max(len(row) for row in main_content) if main_content and any(main_content) else 0
        
        num_compare_rows = len(compare_content) if compare_content else 0
        num_compare_cols = 0
        if compare_content and any(compare_content): # Ensure compare_content itself and its rows are not None
            num_compare_cols = max(len(row) if row is not None else 0 for row in compare_content)

        max_rows = max(num_main_rows, num_compare_rows)
        max_cols = max(num_main_cols, num_compare_cols)

        if max_cols == 0 and max_rows == 0: 
             html_parts.append("<tr><td>Empty table structure.</td></tr>")
        else:
            for i in range(max_rows):
                html_parts.append("<tr>")
                for j in range(max_cols):
                    cell_main_text_orig, cell_main_exists = "", False
                    if i < num_main_rows and main_content[i] is not None and j < len(main_content[i]):
                        cell_main_text_orig, cell_main_exists = str(main_content[i][j]), True
                    
                    cell_compare_text_orig, cell_compare_exists = "", False
                    if compare_content and i < num_compare_rows and compare_content[i] is not None and j < len(compare_content[i]):
                        cell_compare_text_orig, cell_compare_exists = str(compare_content[i][j]), True
                    
                    cell_display_final_content = "" # This will hold the main content part (text, highlighted text, or &nbsp;)
                    cell_class = ""
                    key_html_snippet = ""
                    apply_key_context = False # Flag to determine if key context should be prepended

                    # Determine key_html_snippet if this cell (j) is not the first one in the row.
                    # The key is always sourced from main_content (the table for the current side being rendered).
                    if j > 0: # Only apply key context if it's not the first column
                        if i < num_main_rows and main_content[i] is not None and (j-1) < len(main_content[i]):
                            key_cell_text_from_main = str(main_content[i][j-1])
                            if key_cell_text_from_main.strip(): # Ensure key_cell_text is not just whitespace
                                key_html_snippet = f'<div class="diff-table-key-context"><span class="key-label">{self._escape(key_cell_text_from_main)}:</span></div>'
                    
                    if cell_main_exists:
                        if cell_compare_exists: # Cell exists in both main and compare
                            norm_main = self._normalize_cell_for_render(cell_main_text_orig)
                            norm_compare = self._normalize_cell_for_render(cell_compare_text_orig)
                            if norm_main == norm_compare:
                                cell_class = "cell-similar"
                                cell_display_final_content = self._escape(cell_main_text_orig)
                                # apply_key_context remains False for similar cells (unless changed)
                            else: 
                                cell_class = "cell-modified"
                                highlighted_current, _ = self._highlight_word_diff(cell_main_text_orig, cell_compare_text_orig)
                                cell_display_final_content = highlighted_current
                                apply_key_context = True # Apply key for modified cells
                        else: # Cell exists in main_content, but not in compare_content
                            # This cell's content is part of the current side's table.
                            cell_class = "cell-deleted" if is_left_side else "cell-inserted"
                            cell_display_final_content = self._escape(cell_main_text_orig)
                            apply_key_context = True # Apply key for cells showing existing content that's diff
                    elif cell_compare_exists: # Cell does NOT exist in main_content, but exists in compare_content
                        # This cell is a placeholder on the current side for content from the other side.
                        cell_class = "cell-inserted" if is_left_side else "cell-deleted"
                        cell_display_final_content = "&nbsp;" # Placeholder
                        apply_key_context = True # Apply key context for the placeholder
                    else: # Neither cell exists (e.g., padding for uneven tables, or truly empty cell in both)
                        cell_display_final_content = "&nbsp;"
                        cell_class = "cell-empty" 

                    # Construct the final display HTML for the cell
                    if apply_key_context and key_html_snippet: # Prepend key if applicable and key exists
                        cell_display_text = f'{key_html_snippet}<div class="value-content">{cell_display_final_content}</div>'
                    elif apply_key_context: # Is a diff cell, but no key (e.g. first column)
                         cell_display_text = f'<div class="value-content">{cell_display_final_content}</div>'
                    else: # Not a diff cell where key context is applied (e.g. similar, or empty)
                        cell_display_text = cell_display_final_content

                    html_parts.append(f'<td class="{cell_class}">{cell_display_text}</td>')
                html_parts.append("</tr>")
        html_parts.append("</table>")
        return "".join(html_parts)

    def _generate_paired_table_diff_html(self, 
                                         table1_content: Optional[List[List[str]]], 
                                         table2_content: Optional[List[List[str]]],
                                         table1_id: Optional[str], 
                                         table2_id: Optional[str]) -> str:
        side1_html = self._render_single_html_table_for_pairing(table1_content, table2_content, is_left_side=True)
        side2_html = self._render_single_html_table_for_pairing(table2_content, table1_content, is_left_side=False)

        html = ['<div class="table-diff-container">']
        html.append('<div class="table-diff-side">')
        pdf1_table_name = table1_id if table1_id else "Table"
        html.append(f'<h6>{self._escape(pdf1_table_name)} ({self._escape(self.pdf1_name)})</h6>')
        html.append(side1_html)
        html.append('</div>')
        html.append('<div class="table-diff-separator"></div>') 
        html.append('<div class="table-diff-side">')
        pdf2_table_name = table2_id if table2_id else "Table"
        html.append(f'<h6>{self._escape(pdf2_table_name)} ({self._escape(self.pdf2_name)})</h6>')
        html.append(side2_html)
        html.append('</div>')
        html.append('</div>')
        return "".join(html)

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
        
        for pg_num_str, page_data in results.get("pages", {}).items():
            pg_num = int(pg_num_str)
            raw_text_diff_list = page_data.get("text_differences_raw", page_data.get("text_differences", []))
            raw_table_diff_list = page_data.get("table_differences_raw", page_data.get("table_differences", []))

            processed_pages[pg_num] = {
                "text_differences": self._prepare_text_diffs_for_render(raw_text_diff_list, pg_num),
                "table_differences": self._prepare_table_diffs_for_render(raw_table_diff_list, pg_num)
            }
            pages_processed_count += 1
            if progress_callback and total_pages_to_process > 0:
                progress_callback(0.1 + 0.7 * (pages_processed_count / total_pages_to_process))
        
        summary_stats = self._calculate_summary(results) 
        if progress_callback: progress_callback(0.85)
        
        env = Environment(loader=BaseLoader(), autoescape=select_autoescape(['html', 'xml']))
        env.filters['escapejs'] = lambda s: self._escape(s).replace("'", "\\'")
        
        comparison_datetime_obj = datetime.now()
        comparison_date_formatted = comparison_datetime_obj.strftime("%Y-%m-%d")
        comparison_datetime_formatted = comparison_datetime_obj.strftime("%Y-%m-%d %H:%M:%S")

        tpl = env.from_string(self._TEMPLATE)
        html_output = tpl.render(
            generation_datetime=comparison_datetime_formatted,
            comparison_date=comparison_date_formatted,
            pdf1_name=self._escape(self.pdf1_name), 
            pdf2_name=self._escape(self.pdf2_name),
            pages=processed_pages, 
            summary=summary_stats,
            escape_html=self._escape
        )
        if progress_callback: progress_callback(0.95)
        
        ts = comparison_datetime_obj.strftime("%Y%m%d_%H%M%S")
        safe_pdf1_name = re.sub(r'[^\w\-_.]', '_', pdf1_name)
        safe_pdf2_name = re.sub(r'[^\w\-_.]', '_', pdf2_name)
        outfile_name = f"Comparison_Report_{safe_pdf1_name}_vs_{safe_pdf2_name}_{ts}.html"
        fullpath = os.path.join(self.output_dir, outfile_name)
        
        with open(fullpath, "w", encoding="utf-8") as fh: fh.write(html_output)
        logger.info(f"Report generated: {fullpath}")
        if progress_callback: progress_callback(1.0)
        return fullpath

    def _prepare_text_diffs_for_render(self, raw_diffs: List[Dict], current_page_num: int) -> List[Dict]:
        candidate_items = []
        temp_raw_diffs = []
        for idx, d in enumerate(raw_diffs):
            dc = d.copy()
            if dc["status"] == "deleted" and "id1" not in dc and "id" not in dc : dc["id1"] = f"raw_del_text_{idx}"
            elif dc["status"] == "inserted" and "id2" not in dc and "id" not in dc: dc["id2"] = f"raw_ins_text_{idx}"
            temp_raw_diffs.append(dc)
        raw_diffs = temp_raw_diffs

        candidate_items.extend([d for d in raw_diffs if d["status"] not in ["deleted", "inserted"]])
        
        deleted_items = [d for d in raw_diffs if d["status"] == "deleted"]
        inserted_items = [d for d in raw_diffs if d["status"] == "inserted"]
        consumed_deleted_indices, consumed_inserted_indices = set(), set()

        for del_idx, d_item in enumerate(deleted_items):
            if del_idx in consumed_deleted_indices: continue
            d_text1, d_page1 = str(d_item.get("text1", "")), d_item.get("page1")
            d_item_id1 = d_item.get("id1", d_item.get("id", f"del_text_ref_{del_idx}"))

            best_i_match, best_i_idx, highest_sim = None, -1, -1.0
            for ins_idx, i_item in enumerate(inserted_items):
                if ins_idx in consumed_inserted_indices: continue
                i_text2, i_page2 = str(i_item.get("text2", "")), i_item.get("page2")
                similarity = self._calculate_text_similarity_local(d_text1, i_text2)
                if similarity > highest_sim:
                    highest_sim, best_i_match, best_i_idx = similarity, i_item, ins_idx
            
            if best_i_match and highest_sim >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                consumed_deleted_indices.add(del_idx)
                consumed_inserted_indices.add(best_i_idx)
                i_page2_val = best_i_match.get("page2")
                final_status = "modified" if d_page1 == i_page2_val else "moved"
                i_item_id2 = best_i_match.get("id2", best_i_match.get("id", f"ins_text_ref_{best_i_idx}"))
                paired_item = {
                    "text1": d_text1, "text2": best_i_match.get("text2",""), "score": highest_sim,
                    "page1": d_page1, "page2": i_page2_val,
                    "block1_bbox": d_item.get("block1_bbox"), "block2_bbox": best_i_match.get("block2_bbox"),
                    "status": final_status,
                    "id1": d_item_id1, 
                    "id2": i_item_id2  
                }
                candidate_items.append(paired_item)
        
        for del_idx, d_item in enumerate(deleted_items):
            if del_idx not in consumed_deleted_indices: 
                d_item.setdefault("id1", d_item.get("id", f"del_text_unpaired_{del_idx}"))
                candidate_items.append(d_item)
        for ins_idx, i_item in enumerate(inserted_items):
            if ins_idx not in consumed_inserted_indices: 
                i_item.setdefault("id2", i_item.get("id", f"ins_text_unpaired_{ins_idx}"))
                candidate_items.append(i_item)

        final_rendered_list = []
        def get_sort_key_for_render(item):
            page_val = item.get("page1") if item.get("page1") is not None else item.get("page2", float('inf'))
            bbox = item.get("block1_bbox") if item.get("block1_bbox") else item.get("block2_bbox")
            y_coord = bbox[1] if bbox and len(bbox) > 1 else float('inf')
            x_coord = bbox[0] if bbox and len(bbox) > 0 else float('inf')
            return (page_val, y_coord, x_coord)

        for item in sorted(candidate_items, key=get_sort_key_for_render):
            status = item["status"]
            text1, text2 = str(item.get("text1", "")), str(item.get("text2", ""))
            page1, page2 = item.get("page1"), item.get("page2")
            score = item.get("score", 0.0)
            display_item = item.copy()
            relevant_for_this_page = False

            if status == "moved" and score == 1.0:
                relevant_for_this_page = False
            elif status == "modified":
                if current_page_num == page1: 
                    relevant_for_this_page = True
                    highlighted1, highlighted2 = self._highlight_word_diff(text1, text2)
                    display_item["display_text1"] = highlighted1 
                    display_item["display_text2"] = highlighted2
            elif status == "moved": 
                if current_page_num == page1 or current_page_num == page2: 
                    relevant_for_this_page = True
                    h1, h2 = self._highlight_word_diff(text1, text2)
                    display_item["display_text1"] = h1
                    display_item["display_text2"] = h2
                    display_item["move_info"] = f"Moved: PDF1 p.{page1} ➜ PDF2 p.{page2} (Similarity: {score*100:.1f}%)"
            elif status == "deleted":
                if current_page_num == page1:
                    relevant_for_this_page = True
                    display_item["display_text1"] = self._escape(text1)
                    display_item["display_text2"] = f"<i>Not present in {self._escape(self.pdf2_name)}</i>"
            elif status == "inserted":
                if current_page_num == page2:
                    relevant_for_this_page = True
                    display_item["display_text1"] = f"<i>Not present in {self._escape(self.pdf1_name)}</i>"
                    display_item["display_text2"] = self._escape(text2)
            elif status == "matched": 
                if current_page_num == page1 :
                    relevant_for_this_page = True
                    display_item["display_text1"] = self._escape(text1) 
                    display_item["display_text2"] = self._escape(text2)
            
            if relevant_for_this_page:
                final_rendered_list.append(display_item)
        return final_rendered_list

    def _prepare_table_diffs_for_render(self, raw_diffs: List[Dict], current_page_num: int) -> List[Dict]:
        candidate_items = []
        temp_raw_diffs = []
        for idx, d in enumerate(raw_diffs):
            dc = d.copy()
            if dc["status"] == "deleted" and "id1" not in dc and "id" not in dc and "table1_id" not in dc: dc["id1"] = f"raw_del_tab_{idx}"
            elif dc["status"] == "inserted" and "id2" not in dc and "id" not in dc and "table2_id" not in dc: dc["id2"] = f"raw_ins_tab_{idx}"
            temp_raw_diffs.append(dc)
        raw_diffs = temp_raw_diffs

        candidate_items.extend([d for d in raw_diffs if d["status"] not in ["deleted", "inserted"]])
        deleted_tables = [d for d in raw_diffs if d["status"] == "deleted"]
        inserted_tables = [d for d in raw_diffs if d["status"] == "inserted"]
        consumed_deleted_indices, consumed_inserted_indices = set(), set()

        for del_idx, d_table in enumerate(deleted_tables):
            if del_idx in consumed_deleted_indices: continue
            d_content1, d_page1 = d_table.get("table1_content"), d_table.get("page1")
            d_id1 = d_table.get("table1_id", d_table.get("id1", d_table.get("id", f"del_tab_ref_{del_idx}")))

            best_i_match, best_i_idx, highest_sim = None, -1, -1.0
            for ins_idx, i_table in enumerate(inserted_tables):
                if ins_idx in consumed_inserted_indices: continue
                i_content2 = i_table.get("table2_content")
                sim_score = 0.0
                if d_content1 and i_content2:
                    flat1 = " ".join([" ".join(map(str, r)) for r in d_content1 if r])
                    flat2 = " ".join([" ".join(map(str, r)) for r in i_content2 if r])
                    sim_score = self._calculate_text_similarity_local(flat1, flat2)
                elif d_content1 is None and i_content2 is None: # Both are None means they are similar (empty)
                    sim_score = 1.0

                if sim_score > highest_sim:
                    highest_sim, best_i_match, best_i_idx = sim_score, i_table, ins_idx

            if best_i_match and highest_sim >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                consumed_deleted_indices.add(del_idx)
                consumed_inserted_indices.add(best_i_idx)
                i_page2, i_content2 = best_i_match.get("page2"), best_i_match.get("table2_content")
                i_id2 = best_i_match.get("table2_id", best_i_match.get("id2", best_i_match.get("id", f"ins_tab_ref_{best_i_idx}")))
                new_status = "modified" if d_page1 == i_page2 else "moved"
                candidate_items.append({
                    "status": new_status, "score": highest_sim,
                    "page1": d_page1, "page2": i_page2,
                    "table1_id": d_id1, "table2_id": i_id2,
                    "table1_bbox": d_table.get("table1_bbox"), "table2_bbox": best_i_match.get("table2_bbox"),
                    "table1_content": d_content1, "table2_content": i_content2,
                    "id1": d_id1,
                    "id2": i_id2
                })
        for del_idx, d_table in enumerate(deleted_tables):
            if del_idx not in consumed_deleted_indices:
                d_table.setdefault("id1", d_table.get("table1_id", d_table.get("id", f"del_tab_unpaired_{del_idx}")))
                candidate_items.append(d_table)
        for ins_idx, i_table in enumerate(inserted_tables):
            if ins_idx not in consumed_inserted_indices:
                i_table.setdefault("id2", i_table.get("table2_id", i_table.get("id", f"ins_tab_unpaired_{ins_idx}")))
                candidate_items.append(i_table)

        final_rendered_list = []
        def get_sort_key_for_render_table(item):
            page_val = item.get("page1") if item.get("page1") is not None else item.get("page2", float('inf'))
            bbox = item.get("table1_bbox") if item.get("table1_bbox") else item.get("table2_bbox")
            y_coord = bbox[1] if bbox and len(bbox) > 1 else float('inf')
            x_coord = bbox[0] if bbox and len(bbox) > 0 else float('inf')
            return (page_val, y_coord, x_coord)

        for item in sorted(candidate_items, key=get_sort_key_for_render_table):
            status, p1, p2 = item["status"], item.get("page1"), item.get("page2")
            score = item.get("score", 0.0)
            disp_item, relevant = item.copy(), False

            t1_content = item.get("table1_content")
            t2_content = item.get("table2_content")
            t1_id = item.get("table1_id")
            t2_id = item.get("table2_id")

            if status == "moved" and score == 1.0: # Identical moved tables are less likely to need detailed cell diff
                relevant = False # Or choose to show them if needed
            elif status == "moved":
                if current_page_num == p1 or current_page_num == p2:
                    relevant = True
                    disp_item["move_info"] = f"Moved: PDF1 p.{p1} ➜ PDF2 p.{p2} (Similarity: {score*100:.1f}%)"
                    disp_item["diff_html"] = self._generate_paired_table_diff_html(t1_content, t2_content, t1_id, t2_id)
            elif status == "modified":
                 # MODIFICATION START: Filter out 100% similar modified tables
                 if score == 1.0:
                     relevant = False
                 # MODIFICATION END
                 elif current_page_num == p1: # Show on page of PDF1
                     relevant = True
                     disp_item["diff_html"] = self._generate_paired_table_diff_html(t1_content, t2_content, t1_id, t2_id)
            elif status == "matched":
                 if current_page_num == p1: # Show on page of PDF1
                     relevant = True
                     # For matched, you might choose not to show detailed diff_html or a simplified version
                     disp_item["diff_html"] = self._generate_paired_table_diff_html(t1_content, t2_content, t1_id, t2_id)
            elif status == "deleted":
                 if current_page_num == p1: # Table from PDF1 deleted in PDF2, show on PDF1's page
                     relevant = True
                     disp_item["diff_html"] = self._generate_paired_table_diff_html(t1_content, None, t1_id, None)
            elif status == "inserted":
                 if current_page_num == p2: # Table inserted in PDF2, show on PDF2's page
                     relevant = True
                     disp_item["diff_html"] = self._generate_paired_table_diff_html(None, t2_content, None, t2_id)

            if relevant: final_rendered_list.append(disp_item)
        return final_rendered_list
    def _calculate_summary(self, results: Dict) -> Dict:
        total_pages_in_docs = results.get("max_pages", 0)
        pages_with_diff_count = 0
        
        s_modifications = 0
        s_insertions = 0
        s_deletions = 0
        s_moved_elements = 0
        
        processed_unique_ids_for_dashboard = set() 

        for pg_num_str, page_data in results.get("pages", {}).items():
            current_page_num = int(pg_num_str)
            page_has_reportable_item_this_iteration = False

            raw_text_diffs_for_page = page_data.get("text_differences_raw", page_data.get("text_differences", []))
            text_items_for_page = self._prepare_text_diffs_for_render(raw_text_diffs_for_page, current_page_num)
            
            if text_items_for_page: page_has_reportable_item_this_iteration = True
            for item in text_items_for_page: 
                temp_id1 = item.get("id1") 
                uid_part1 = str(temp_id1) if temp_id1 is not None else \
                                str(item.get("block1_bbox", "_NO_BBOX1_")) + str(item.get("text1", "")[:20])

                temp_id2 = item.get("id2")
                uid_part2 = str(temp_id2) if temp_id2 is not None else \
                                str(item.get("block2_bbox", "_NO_BBOX2_")) + str(item.get("text2", "")[:20])
                
                item_key_for_dashboard = None
                current_status = item["status"] 

                if current_status == "modified" or (current_status == "moved" and item.get("score", 0.0) < 1.0) : 
                    item_key_for_dashboard = ("text", tuple(sorted((uid_part1, uid_part2))), current_status)
                elif current_status == "deleted":
                    item_key_for_dashboard = ("text", uid_part1, current_status)
                elif current_status == "inserted":
                     item_key_for_dashboard = ("text", uid_part2, current_status)
                elif current_status == "moved" and item.get("score", 0.0) == 1.0 : # perfectly moved, count as moved if not already
                    item_key_for_dashboard = ("text", tuple(sorted((uid_part1, uid_part2))), "perfectly_moved_text")


                if item_key_for_dashboard and item_key_for_dashboard not in processed_unique_ids_for_dashboard:
                    processed_unique_ids_for_dashboard.add(item_key_for_dashboard)
                    if current_status == "modified": s_modifications += 1
                    elif current_status == "inserted": s_insertions += 1
                    elif current_status == "deleted": s_deletions += 1
                    elif current_status == "moved": s_moved_elements +=1


            raw_table_diffs_for_page = page_data.get("table_differences_raw", page_data.get("table_differences", []))
            table_items_for_page = self._prepare_table_diffs_for_render(raw_table_diffs_for_page, current_page_num)

            if table_items_for_page: page_has_reportable_item_this_iteration = True
            for item in table_items_for_page:
                temp_id1 = item.get("id1", item.get("table1_id"))
                uid_part1 = str(temp_id1) if temp_id1 is not None else \
                                str(item.get("table1_bbox", "_NO_TBBOX1_")) + str(item.get("table1_content", [])[:1])


                temp_id2 = item.get("id2", item.get("table2_id"))
                uid_part2 = str(temp_id2) if temp_id2 is not None else \
                                str(item.get("table2_bbox", "_NO_TBBOX2_")) + str(item.get("table2_content", [])[:1])

                item_key_for_dashboard = None
                current_status = item["status"]
                if current_status == "modified" or (current_status == "moved" and item.get("score", 0.0) < 1.0):
                    item_key_for_dashboard = ("table", tuple(sorted((uid_part1, uid_part2))), current_status)
                elif current_status == "deleted":
                    item_key_for_dashboard = ("table", uid_part1, current_status)
                elif current_status == "inserted":
                     item_key_for_dashboard = ("table", uid_part2, current_status)
                elif current_status == "moved" and item.get("score", 0.0) == 1.0:
                     item_key_for_dashboard = ("table", tuple(sorted((uid_part1, uid_part2))), "perfectly_moved_table")


                if item_key_for_dashboard and item_key_for_dashboard not in processed_unique_ids_for_dashboard:
                    processed_unique_ids_for_dashboard.add(item_key_for_dashboard)
                    if current_status == "modified": s_modifications += 1
                    elif current_status == "inserted": s_insertions += 1
                    elif current_status == "deleted": s_deletions += 1
                    elif current_status == "moved": s_moved_elements += 1
            
            if page_has_reportable_item_this_iteration:
                pages_with_diff_count +=1
        
        text_diff_blocks_count = sum(1 for uid_tuple in processed_unique_ids_for_dashboard if uid_tuple[0] == "text" and uid_tuple[2] != "perfectly_moved_text")
        table_diff_blocks_count = sum(1 for uid_tuple in processed_unique_ids_for_dashboard if uid_tuple[0] == "table" and uid_tuple[2] != "perfectly_moved_table")
        # s_moved_elements is already counted from items with status "moved"

        return {
            "total_pages": total_pages_in_docs, 
            "pages_with_differences": pages_with_diff_count,
            "text_differences": text_diff_blocks_count, # Excludes perfectly moved from this specific count
            "table_differences": table_diff_blocks_count, # Excludes perfectly moved from this specific count
            "ds_modifications": s_modifications,
            "ds_insertions": s_insertions,
            "ds_deletions": s_deletions,
            "ds_moved_elements": s_moved_elements, # This now correctly sums up all "moved" identified
        }

    _TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDF Comparison: {{ pdf1_name }} vs {{ pdf2_name }}</title>
<style>
body{font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;background:#f0f2f5;margin:0;padding:20px;color:#333;line-height:1.6}
.container{max-width:1300px;margin:20px auto;background:#fff;padding:25px;box-shadow:0 2px 15px rgba(0,0,0,.1); border-radius: 8px;}

header{background:#BF3F3F; color:#fff; padding:20px 25px; margin:-25px -25px 20px -25px; border-bottom: 5px solid #A93434; border-radius: 8px 8px 0 0;}
.report-title-line { font-size: 26px; font-weight: bold; margin-bottom: 4px;}
.comparison-files-line { font-size: 16px; margin-bottom: 8px;}
.comparison-files-line b { font-weight: 600;}
.generated-on-line { font-size: 13px; color: #f0f0f0; opacity:0.9;}

.report-info-grid { display: grid; grid-template-columns: max-content 1fr; gap: 8px 20px; margin-bottom: 25px; background-color: #f8f9fa; padding: 15px 20px; border-radius: 6px; border: 1px solid #e9ecef;}
.report-info-grid > div { display: contents; }
.report-info-grid > div > span:first-child { font-weight: 600; color: #495057; }
.report-info-grid > div > span:last-child { color: #212529; }

/* Refined Summary Dashboard (for Modifications, Insertions, etc.) */
.summary-dashboard { 
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); /* Responsive columns */
    gap: 18px; /* Spacing between tiles */
    margin: 25px 0; 
    padding: 0; /* No padding on the container itself */
}
.dashboard-item { 
    background: #ffffff; 
    border: 1px solid #e0e0e0; /* Lighter border */
    border-radius: 6px; 
    padding: 18px; 
    text-align: center; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.07); /* Subtle shadow */
    transition: transform 0.2s ease-out, box-shadow 0.2s ease-out;
}
.dashboard-item:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
.dashboard-item .dashboard-value { 
    font-size: 2em; 
    font-weight: 700; 
    margin-bottom: 6px; 
    line-height: 1.1;
}
.dashboard-item .dashboard-label { 
    font-size: 0.85em; 
    color: #555; 
    font-weight: 600; 
    text-transform: uppercase;
}
.dashboard-item.modifications .dashboard-value { color: #ffab00; } 
.dashboard-item.insertions .dashboard-value { color: #00c853; } 
.dashboard-item.deletions .dashboard-value { color: #dd2c00; } 
.dashboard-item.moved-elements .dashboard-value { color: #2962ff; } 

/* ---- Primary Summary Tile Look (as per screenshot) ---- */
.summary {
    display: grid; 
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); 
    gap: 20px;
    margin: 30px 0; 
    padding: 0;
}
.stat-item {
    background: #fff; 
    border-radius: 8px; 
    padding: 20px;
    text-align: left; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    position: relative; 
    overflow: hidden; 
    border-left-width: 6px; 
    border-left-style: solid;
}
.stat-item.total-pages { border-left-color: #4CAF50; } 
.stat-item.pages-diff { border-left-color: #F44336; } 
.stat-item.text-diff { border-left-color: #FFC107; }  
.stat-item.table-diff { border-left-color: #FF9800; } 

.stat-item .stat-label {
    font-size: 0.9em; 
    color: #555; 
    margin-bottom: 8px;
    text-transform: uppercase;
    font-weight: 600;
    display: block; 
}
.stat-item .stat-value {
    font-size: 2.8em; 
    font-weight: 700; 
    line-height: 1;
    display: block; 
}
.stat-item.total-pages .stat-value { color: #4CAF50; }
.stat-item.pages-diff .stat-value { color: #F44336; }
.stat-item.text-diff .stat-value { color: #FFC107; }
.stat-item.table-diff .stat-value { color: #FF9800; }
/* ---- End of Primary Summary Tile Look ---- */


.diff-legend { display: flex; flex-wrap: wrap; gap: 10px; padding: 12px 15px; background: #f8f9fa; border-radius: 6px; margin: 25px 0; border: 1px solid #e9ecef; }
.legend-item { display: inline-block; padding: 6px 12px; border-radius: 4px; font-size: 0.875rem; font-weight: 500; text-align: center; }
.legend-deleted { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
.legend-inserted { background-color: #cce5ff; color: #004085; border: 1px solid #b8daff;}
.legend-modified { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;}
.legend-matched { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
.legend-moved { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb;}

.page-nav{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:25px;position:sticky;top:0;background:rgba(255,255,255,0.98);padding:10px 0;z-index:1000;border-bottom:1px solid #dee2e6; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
.page-button{padding:8px 15px;background:#f8f9fa;border:1px solid #ced4da; border-radius:4px;text-decoration:none;color:#333; /* Unhighlighted color to dark grey/black */ font-weight:500; transition: all 0.2s ease-in-out;}
.page-button:hover{background:#BF3F3F; color:#fff; border-color:#A93434;} /* Hover to match active style */
.page-button.active{background:#BF3F3F;color:#fff;border-color:#A93434; font-weight:bold;} 

.page-section{margin-bottom:25px;border:1px solid #dee2e6;border-radius:6px;padding:20px; background: #fff;}
.page-header{background:#f8f9fa;padding:12px 18px;margin:-20px -20px 20px -20px;border-bottom:1px solid #dee2e6;display:flex;justify-content:space-between;align-items:center;border-radius:6px 6px 0 0;}
.page-title{color:#333; font-size: 1.5em; font-weight: 600;} /* Page title to dark grey/black */

.difference-item{margin-bottom:20px;border:1px solid #e0e0e0;border-radius:5px;overflow:hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.04);}
.difference-header{padding:10px 15px;background:#f8f9fa;border-bottom:1px solid #e0e0e0;font-weight:600;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap; gap: 10px;}
.difference-content{display:flex;width:100%}
.difference-side{flex:1;padding:15px;border-right:1px solid #e9ecef;overflow-x:auto; min-width: 45%; background: #fff;}
.difference-side:last-child{border-right:none}
.side-header{font-weight:600;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #f1f1f1; color: #343a40;}

.status-badge{display:inline-block;padding:4px 8px;border-radius:4px;font-size:13px;font-weight:600;text-transform:uppercase; letter-spacing: 0.5px;}
.status-matched{background:#d4edda;color:#155724; border: 1px solid #c3e6cb;}
.status-modified{background:#fff3cd;color:#856404; border: 1px solid #ffeeba;}
.status-deleted{background:#f8d7da;color:#721c24; border: 1px solid #f5c6cb;}
.status-inserted{background:#cce5ff;color:#004085; border: 1px solid #b8daff;}
.status-moved{background:#d1ecf1;color:#0c5460; border: 1px solid #bee5eb;}
.move-info{font-style:italic;color:#555;margin-left:auto; font-weight:normal; font-size:0.9em;}
.move-info em, .move-info i {font-style:normal;font-weight:bold;color:#BF3F3F;}

.text-deleted, .text-inserted, .text-modified, .text-matched { padding: 5px; white-space: pre-wrap; word-wrap: break-word; font-size: 0.95em; line-height: 1.5;}
.text-deleted { background:#FFEDED; color:#721C24; }
.text-inserted { background:#EDF5FF; color:#004085; }
.text-modified { /* Base style */ } 
.text-matched { background:#f0fff0; }

.diff-word-deleted { background-color: #ffcdd2; color: #b71c1c; padding: 1px 2px; border-radius: 3px; font-weight: 500;}
.diff-word-inserted { color: red !important; font-weight: bold !important; background-color: transparent !important; padding: 1px 0px; }

.table-diff-container{display:flex;width:100%; gap:1px; background-color: #e9ecef;}
.table-diff-side{flex:1;padding:10px;overflow-x:auto; background: #fff;}
.table-diff-side h6 { margin-top:0; margin-bottom: 10px; font-size: 1.05em; color: #343a40;}

.diff-table{width:100%;border-collapse:collapse;font-size:13px;table-layout:auto;}
.diff-table th,.diff-table td{border:1px solid #dee2e6;padding:8px 10px;text-align:left;word-wrap:break-word; vertical-align: top;}
.diff-table th{background:#f8f9fa;font-weight:600; color: #495057;}
.diff-table-key-context { margin-bottom: 4px; } /* Styles the div holding the key */
.diff-table-key-context .key-label { font-weight: bold; color: #555; font-size:0.9em; } /* Styles the key text itself */
.value-content { /* wrapper for the actual value, ensures it's on a new line after key if key exists */ }

.cell-similar{background:#f0fff0 !important;}
.cell-modified { background:#fff9e6 !important; }
.cell-deleted{background:#ffebee !important;}
.cell-inserted{background:#e3f2fd !important;}
.cell-empty { background: #fdfdfd; }

@media(max-width:992px){ 
    .summary-dashboard, .summary { flex-direction: column; }
    .dashboard-item, .stat-item { margin-bottom: 15px;}
}
@media(max-width:768px){
    .difference-content,.table-diff-container{flex-direction:column}
    .difference-side,.table-diff-side{border-right:none;border-bottom:1px solid #e9ecef;margin-bottom:10px; min-width: 100%;}
    .difference-side:last-child,.table-diff-side:last-child{border-bottom:none;margin-bottom:0}
    .page-title {font-size: 1.3em;}
    .report-info-grid { grid-template-columns: max-content 1fr; }
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function () {
    const pageButtons = document.querySelectorAll('.page-button');
    const pageSections = document.querySelectorAll('.page-section');
    let currentActiveButton = null;
    let isClickScroll = false; 

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

    function updateActiveStateFromHash(hash) {
        const targetButton = document.querySelector(`.page-button[href="${hash}"]`);
        setActiveButton(targetButton);
    }

    if (window.location.hash) {
        updateActiveStateFromHash(window.location.hash);
    } else if (pageButtons.length > 0) {
        setActiveButton(pageButtons[0]);
    }

    window.addEventListener('hashchange', function () {
        if (!isClickScroll) { 
            updateActiveStateFromHash(window.location.hash);
        }
    });

    if ('IntersectionObserver' in window) {
        const observerOptions = {
            root: null,
            rootMargin: '0px 0px -70% 0px', 
            threshold: 0.01 
        };

        const observerCallback = (entries, observer) => {
            if (isClickScroll) return; 

            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const sectionId = entry.target.id;
                    const targetButton = document.querySelector(`.page-button[href="#${sectionId}"]`);
                    setActiveButton(targetButton);
                }
            });
        };
        const observer = new IntersectionObserver(observerCallback, observerOptions);
        pageSections.forEach(section => observer.observe(section));
    }

    pageButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            isClickScroll = true; 

            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);

            setActiveButton(this); 

            if (targetSection) {
                const headerOffsetEl = document.querySelector('.page-nav');
                const headerOffset = headerOffsetEl ? headerOffsetEl.offsetHeight + 15 : 70; 
                
                const elementPosition = targetSection.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth"
                });

                if(history.pushState) {
                    history.pushState(null, null, targetId);
                } else {
                    window.location.hash = targetId;
                }
            }
            setTimeout(() => { isClickScroll = false; }, 1000); 
        });
    });
});
</script>
</head>
<body>
<div class="container">
<header>
  <div class="report-title-line">PDF Comparison Report</div>
  <div class="comparison-files-line">Comparing: <b>{{ pdf1_name }}</b> vs <b>{{ pdf2_name }}</b></div>
  <div class="generated-on-line">Generated on: {{ generation_datetime }}</div>
</header>

<div class="report-info-grid">
    <div><span>Document 1</span><span>{{ pdf1_name }}</span></div>
    <div><span>Document 2</span><span>{{ pdf2_name }}</span></div>
    <div><span>Original FileNames</span><span>{{ pdf1_name }}, {{ pdf2_name }}</span></div>
    <div><span>Comparison Date</span><span>{{ comparison_date }}</span></div>
</div>

<div class="summary">
  <div class="stat-item total-pages"><div class="stat-label">TOTAL PAGES</div><div class="stat-value">{{ summary.total_pages }}</div></div>
  <div class="stat-item pages-diff"><div class="stat-label">PAGES WITH DIFFERENCES</div><div class="stat-value">{{ summary.pages_with_differences }}</div></div>
  <div class="stat-item text-diff"><div class="stat-label">TEXT DIFFERENCE BLOCKS</div><div class="stat-value">{{ summary.text_differences }}</div></div>
  <div class="stat-item table-diff"><div class="stat-label">TABLE DIFFERENCE BLOCKS</div><div class="stat-value">{{ summary.table_differences }}</div></div>
</div>

<div class="summary-dashboard">
  <div class="dashboard-item modifications">
    <div class="dashboard-value">{{ summary.ds_modifications }}</div>
    <div class="dashboard-label">Total Modifications</div>
  </div>
  <div class="dashboard-item insertions">
    <div class="dashboard-value">{{ summary.ds_insertions }}</div>
    <div class="dashboard-label">Insertions</div>
  </div>
  <div class="dashboard-item deletions">
    <div class="dashboard-value">{{ summary.ds_deletions }}</div>
    <div class="dashboard-label">Deletions</div>
  </div>
  <div class="dashboard-item moved-elements">
    <div class="dashboard-value">{{ summary.ds_moved_elements }}</div>
    <div class="dashboard-label">Moved Elements</div>
  </div>
</div>

<div class="diff-legend">
  <span class="legend-item legend-deleted">Deleted (PDF1 only)</span>
  <span class="legend-item legend-inserted">Inserted (PDF2 only)</span>
  <span class="legend-item legend-modified">Modified</span>
  <span class="legend-item legend-matched">Similar / Matched</span>
  <span class="legend-item legend-moved">Moved</span>
</div>

<div class="page-nav">
  {% for pg_num_key in pages.keys()|sort %}
    <a href="#page-{{ pg_num_key }}" class="page-button">Page {{ pg_num_key }}</a>
  {% endfor %}
</div>

{% for pg_num, page_content in pages.items()|sort %}
<section id="page-{{ pg_num }}" class="page-section">
  <div class="page-header">
    <h2 class="page-title">Page {{ pg_num }}</h2>
    {% set has_diffs = page_content.text_differences|length > 0 or page_content.table_differences|length > 0 %}
    {% if has_diffs %}
      <span class="status-badge status-modified">Differences Found</span>
    {% else %}
      <span class="status-badge status-matched">No Differences</span>
    {% endif %}
  </div>

  {% if page_content.text_differences %}
    <h3>Text Differences ({{ page_content.text_differences|length }})</h3>
    {% for d in page_content.text_differences %}
    <div class="difference-item">
      <div class="difference-header">
        <span class="status-badge status-{{ d.status }}">{{ d.status|replace('_', ' ')|title }}</span>
        {% if d.score is defined and (d.status == 'modified' or d.status == 'matched' or (d.status == 'moved' and d.score < 1.0)) %}
          <span>Similarity: {{ "%.1f"|format(d.score*100) }}%</span>
        {% endif %}
        {% if d.move_info %}<span class="move-info">{{ d.move_info|safe }}</span>{% endif %}
      </div>
      <div class="difference-content">
        <div class="difference-side">
          <div class="side-header">{{ pdf1_name }} (Page {{ d.page1 if d.page1 else 'N/A' }})</div>
          <div class="text-content-area text-{{ d.status }}">{{ d.display_text1|safe }}</div>
        </div>
        <div class="difference-side">
          <div class="side-header">{{ pdf2_name }} (Page {{ d.page2 if d.page2 else 'N/A' }})</div>
          <div class="text-content-area text-{{ d.status }}">{{ d.display_text2|safe }}</div>
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
        <span class="status-badge status-{{ d.status }}">{{ d.status|replace('_', ' ')|title }}</span>
        {% if d.score is defined and (d.status == 'modified' or d.status == 'matched' or (d.status == 'moved' and d.score < 1.0)) %}
          <span>Similarity: {{ "%.1f"|format(d.score*100) }}%</span>
        {% endif %}
        {% if d.move_info %}<span class="move-info">{{ d.move_info|safe }}</span>{% endif %}
      </div>
      <div class="difference-content table-content-area">
        {{ d.diff_html|safe }}
      </div>
    </div>
    {% endfor %}
  {% endif %}
  
  {% if not page_content.text_differences and not page_content.table_differences %}
    <p>No differences found on this page.</p>
  {% endif %}
</section>
{% endfor %}
</div>
</body>
</html>
"""
