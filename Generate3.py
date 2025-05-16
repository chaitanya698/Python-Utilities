
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
                                # For the current side being rendered (main_content), show its diff against the other side.
                                if is_left_side: # main_content is from PDF1, compare_content is from PDF2
                                    highlighted_main, _ = self._highlight_word_diff(cell_main_text_orig, cell_compare_text_orig)
                                    cell_display_text = highlighted_main
                                else: # main_content is from PDF2, compare_content is from PDF1
                                    _, highlighted_compare = self._highlight_word_diff(cell_compare_text_orig, cell_main_text_orig) # Note order swap for highlighting
                                    cell_display_text = highlighted_compare
                        else: # Exists in main_content, not in compare_content
                            cell_class = "cell-deleted" if is_left_side else "cell-inserted" 
                            cell_display_text = self._escape(cell_main_text_orig)
                    elif cell_compare_exists: # Exists in compare_content, not in main_content
                        cell_class = "cell-inserted" if is_left_side else "cell-deleted"
                        cell_display_text = "&nbsp;" # Placeholder for alignment on this side
                    else: 
                        cell_display_text = "&nbsp;"
                        cell_class = "cell-empty" 

                    html_parts.append(f'<td class="{cell_class}">{cell_display_text}</td>') # HTML is already handled by _highlight_word_diff
                html_parts.append("</tr>")
        html_parts.append("</table>")
        return "".join(html_parts)

    def _generate_paired_table_diff_html(self, 
                                         table1_content: Optional[List[List[str]]], 
                                         table2_content: Optional[List[List[str]]],
                                         table1_id: Optional[str], 
                                         table2_id: Optional[str]) -> str:
        html = ['<div class="table-diff-container" style="display: flex; width: 100%;">']
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        pdf1_table_name = table1_id if table1_id else "Table from PDF 1"
        html.append(f'<h6>{self._escape(pdf1_table_name)} ({self.pdf1_name})</h6>')
        html.append(self._render_single_html_table_for_pairing(table1_content, table2_content, is_left_side=True))
        html.append('</div>')
        html.append('<div style="width: 1px; background-color: #ccc;"></div>') 
        html.append('<div class="table-diff-side" style="flex: 1; padding: 5px; overflow-x: auto;">')
        pdf2_table_name = table2_id if table2_id else "Table from PDF 2"
        html.append(f'<h6>{self._escape(pdf2_table_name)} ({self.pdf2_name})</h6>')
        html.append(self._render_single_html_table_for_pairing(table2_content, table1_content, is_left_side=False))
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
        for pg_num, page_data in results.get("pages", {}).items():
            processed_pages[pg_num] = {
                "text_differences": self._prepare_text_diffs_for_render(page_data.get("text_differences", []), pg_num),
                "table_differences": self._prepare_table_diffs_for_render(page_data.get("table_differences", []), pg_num)
            }
            pages_processed_count += 1
            if progress_callback and total_pages_to_process > 0:
                progress_callback(0.1 + 0.8 * (pages_processed_count / total_pages_to_process))
        
        summary_stats = self._calculate_summary(results) 
        
        env = Environment(loader=BaseLoader(), autoescape=select_autoescape(['html', 'xml']))
        env.filters['escapejs'] = lambda s: self._escape(s).replace("'", "\\'")
        tpl = env.from_string(self._TEMPLATE)
        html_output = tpl.render(
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pdf1_name=self.pdf1_name, pdf2_name=self.pdf2_name,
            pages=processed_pages, meta=metadata or {}, summary=summary_stats,
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

    def _prepare_text_diffs_for_render(self, diffs: List[Dict], current_page_num: int) -> List[Dict]:
        pre_classified_items = [d for d in diffs if d["status"] not in ["deleted", "inserted"]]
        deleted_items = [d for d in diffs if d["status"] == "deleted"]
        inserted_items = [d for d in diffs if d["status"] == "inserted"]
        candidate_diffs_for_page = list(pre_classified_items)
        consumed_deleted_indices, consumed_inserted_indices = set(), set()

        for del_idx, d_item in enumerate(deleted_items):
            if del_idx in consumed_deleted_indices: continue
            d_text1, d_page1 = d_item.get("text1", ""), d_item.get("page1")
            best_i_match, best_i_idx, highest_sim = None, -1, -1.0
            for ins_idx, i_item in enumerate(inserted_items):
                if ins_idx in consumed_inserted_indices: continue
                i_text2 = i_item.get("text2", "")
                similarity = self._calculate_text_similarity_local(d_text1, i_text2)
                if similarity > highest_sim:
                    highest_sim, best_i_match, best_i_idx = similarity, i_item, ins_idx
            if best_i_match and highest_sim >= self.SIMILARITY_THRESHOLD_FOR_PAIRING:
                consumed_deleted_indices.add(del_idx)
                consumed_inserted_indices.add(best_i_idx)
                i_page2 = best_i_match.get("page2")
                paired_item = {
                    "text1": d_text1, "text2": best_i_match.get("text2", ""), "score": highest_sim,
                    "page1": d_page1, "page2": i_page2,
                    "block1_bbox": d_item.get("block1_bbox"), "block2_bbox": best_i_match.get("block2_bbox"),
                    "status": "modified" if d_page1 == i_page2 else "moved"
                }
                candidate_diffs_for_page.append(paired_item)
        
        for del_idx, d_item in enumerate(deleted_items):
            if del_idx not in consumed_deleted_indices: candidate_diffs_for_page.append(d_item)
        for ins_idx, i_item in enumerate(inserted_items):
            if ins_idx not in consumed_inserted_indices: candidate_diffs_for_page.append(i_item)

        def get_sort_key(item):
            b1, b2 = item.get("block1_bbox"), item.get("block2_bbox")
            y, x = float('inf'), float('inf')
            if current_page_num == item.get("page1") and b1: y,x = b1[1],b1[0]
            elif current_page_num == item.get("page2") and b2: y,x = b2[1],b2[0]
            elif b1: y,x = b1[1],b1[0]
            elif b2: y,x = b2[1],b2[0]
            return (y,x)
        candidate_diffs_for_page.sort(key=get_sort_key)
        
        final_rendered_list = []
        for item in candidate_diffs_for_page:
            status = item["status"]
            text1, text2 = item.get("text1", ""), item.get("text2", "")
            page1, page2 = item.get("page1"), item.get("page2")
            score = item.get("score", 0.0)
            display_item, relevant = item.copy(), False

            if status == "modified":
                if current_page_num == page1 or (page1 is None and current_page_num == page2):
                    relevant = True
                    highlighted1, highlighted2 = self._highlight_word_diff(text1, text2)
                    display_item["display_text1"] = highlighted1 
                    display_item["display_text2"] = highlighted2 
            elif status == "moved":
                if current_page_num == page1 or current_page_num == page2: 
                    relevant = True
                    if score < 1.0 and score >= self.SIMILARITY_THRESHOLD_FOR_PAIRING: 
                        h1, h2 = self._highlight_word_diff(text1, text2)
                        display_item["display_text1"] = h1
                        display_item["display_text2"] = h2
                    else: 
                        display_item["display_text1"] = self._escape(text1)
                        display_item["display_text2"] = self._escape(text2)
                    display_item["move_info"] = f"Moved: {self.pdf1_name} p. {page1} ➜ {self.pdf2_name} p. {page2} (Similarity: {score*100:.2f}%)"
            elif status == "deleted":
                if current_page_num == page1:
                    relevant = True
                    display_item["display_text1"] = self._escape(text1)
                    display_item["display_text2"] = f"<i>Not present in {self.pdf2_name}</i>"
            elif status == "inserted":
                if current_page_num == page2:
                    relevant = True
                    display_item["display_text1"] = f"<i>Not present in {self.pdf1_name}</i>"
                    display_item["display_text2"] = self._escape(text2)
            elif status == "matched": 
                if current_page_num == page1 or (page1 is None and current_page_num == page2): 
                    relevant = True
                    display_item["display_text1"] = self._escape(text1) 
                    display_item["display_text2"] = self._escape(text2)
            
            if relevant: final_rendered_list.append(display_item)
        return final_rendered_list

    def _prepare_table_diffs_for_render(self, diffs: List[Dict], current_page_num: int) -> List[Dict]:
        pre_classified_items = [d for d in diffs if d["status"] not in ["deleted", "inserted"]]
        deleted_tables = [d for d in diffs if d["status"] == "deleted"]
        inserted_tables = [d for d in diffs if d["status"] == "inserted"]
        candidate_table_diffs = list(pre_classified_items)
        consumed_deleted_table_indices, consumed_inserted_table_indices = set(), set()

        for del_idx, d_table in enumerate(deleted_tables):
            if del_idx in consumed_deleted_table_indices: continue
            d_content1, d_page1, d_id1 = d_table.get("table1_content"), d_table.get("page1"), d_table.get("table1_id")
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
                new_status = "modified" if d_page1 == i_page2 else "moved"
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
            if current_page_num==item.get("page1") and b1:y,x=b1[1],b1[0]
            elif current_page_num==item.get("page2") and b2:y,x=b2[1],b2[0]
            elif b1:y,x=b1[1],b1[0]
            elif b2:y,x=b2[1],b2[0]
            return(y,x)
        candidate_table_diffs.sort(key=get_table_sort_key)
        
        final_rendered_list = []
        for item in candidate_table_diffs:
            status,p1,p2=item["status"],item.get("page1"),item.get("page2")
            disp_item,relevant=item.copy(),False
            if status=="moved":
                if current_page_num==p1 or current_page_num==p2:
                    relevant=True
                    disp_item["move_info"]=f"Moved: {self.pdf1_name} p. {p1} ➜ {self.pdf2_name} p. {p2}"
            elif status in["modified","matched"]:
                 if current_page_num==p1 or(p1 is None and current_page_num==p2):relevant=True
            elif status=="deleted":
                 if current_page_num==p1:relevant=True
            elif status=="inserted":
                 if current_page_num==p2:relevant=True
            if relevant:final_rendered_list.append(disp_item)
        return final_rendered_list

    def _calculate_summary(self, results: Dict) -> Dict:
        total_pages_in_docs = results.get("max_pages", 0)
        pages_with_diff = 0
        unique_text_diff_keys, unique_table_diff_keys = set(), set()

        for pg_num, page_data in results.get("pages", {}).items():
            page_has_reportable_diff = False
            for diff in page_data.get("text_differences", []): 
                if diff.get("status") != "matched" or (diff.get("status") == "matched" and diff.get("score", 1.0) < 1.0):
                    page_has_reportable_diff = True
                    status_val, b1_val, b2_val = diff.get("status"), diff.get("block1"), diff.get("block2")
                    p1_val, p2_val = diff.get("page1"), diff.get("page2")
                    key_parts = [
                        str(status_val if status_val is not None else "s_unknown"),
                        str(b1_val.get("hash") if isinstance(b1_val,dict) and b1_val.get("hash") is not None else "h1_none"),
                        str(b2_val.get("hash") if isinstance(b2_val,dict) and b2_val.get("hash") is not None else "h2_none"),
                        str(p1_val if p1_val is not None else "p1_none"), 
                        str(p2_val if p2_val is not None else "p2_none")  
                    ]
                    unique_text_diff_keys.add(tuple(sorted(key_parts)))
            for diff in page_data.get("table_differences", []):
                 if diff.get("status") != "matched" or (diff.get("status") == "matched" and diff.get("score", 1.0) < 1.0):
                    page_has_reportable_diff = True
                    s_val, t1id_val, t2id_val = diff.get("status"), diff.get("table1_id"), diff.get("table2_id")
                    p1_val, p2_val = diff.get("page1"), diff.get("page2")
                    key_parts = [
                        str(s_val if s_val is not None else "s_unknown_tbl"), 
                        str(t1id_val if t1id_val is not None else "id1_none_tbl"),
                        str(t2id_val if t2id_val is not None else "id2_none_tbl"),
                        str(p1_val if p1_val is not None else "p1_none_tbl"),
                        str(p2_val if p2_val is not None else "p2_none_tbl")
                    ]
                    unique_table_diff_keys.add(tuple(sorted(key_parts)))
            if page_has_reportable_diff: pages_with_diff +=1
        
        return {
            "total_pages": total_pages_in_docs, "pages_with_differences": pages_with_diff,
            "text_differences": len(unique_text_diff_keys), "table_differences": len(unique_table_diff_keys),
        }

    _TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PDF Comparison: {{ pdf1_name }} vs {{ pdf2_name }}</title>
<style>
body{font-family:Arial,Helvetica,sans-serif;background:#f5f5f5;margin:0;padding:20px;color:#333;line-height:1.6}
.container{max-width:1200px;margin:0 auto;background:#fff;padding:20px;box-shadow:0 0 10px rgba(0,0,0,.1)}
header{background:#4285F4;color:#fff;padding:15px;margin:-20px -20px 20px -20px}
h1,h2,h3{margin:0 0 10px}
.summary{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:15px;margin:20px 0}
.stat-item{background:#f8f9fa;border-radius:4px;padding:10px;text-align:center}
.stat-value{font-size:20px;font-weight:700;color:#4285F4;margin:5px 0}
.stat-label{font-size:14px;color:#666}
.diff-legend{display:flex;flex-wrap:wrap;gap:10px;padding:12px;background:#f8f8f8;border-radius:5px;margin:15px 0}
.legend-item{display:flex;align-items:center;font-size:.9rem}
.legend-color{width:16px;height:16px;border-radius:3px;margin-right:6px}
.page-nav{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:20px;position:sticky;top:0;background:#fff;padding:10px 0;z-index:100;border-bottom:1px solid #eee}
.page-button{padding:8px 12px;background:#f1f1f1;border:1px solid #ddd;border-bottom:none;border-radius:4px 4px 0 0;text-decoration:none;color:#333}
.page-button:hover{background:#e0e0e0}
.page-button.active{background:#4285F4;color:#fff;border-color:#4285F4}
.page-section{margin-bottom:20px;border:1px solid #eee;border-radius:4px;padding:15px}
.page-header{background:#f8f9fa;padding:10px;margin:-15px -15px 15px -15px;border-bottom:1px solid #eee;display:flex;justify-content:space-between;align-items:center}
.page-title{color:#4285F4}
.difference-item{margin-bottom:15px;border:1px solid #eee;border-radius:4px;overflow:hidden}
.difference-header{padding:8px 12px;background:#f8f9fa;border-bottom:1px solid #eee;font-weight:700;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap; gap: 10px;}
.difference-content{display:flex;width:100%}
.difference-side{flex:1;padding:10px;border-right:1px solid #eee;overflow-x:auto; min-width: 45%;}
.difference-side:last-child{border-right:none}
.side-header{font-weight:700;margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #eee}
.status-badge{display:inline-block;padding:3px 6px;border-radius:3px;font-size:12px;font-weight:700;text-transform:uppercase}
.status-matched{background:#D4EDDA;color:#155724}
.status-modified{background:#FCF8E3;color:#856404}
.status-deleted{background:#F8D7DA;color:#721C24}
.status-inserted{background:#CCE5FF;color:#004085}
.status-moved{background:#D1ECF1;color:#0C5460}
.move-info{font-style:italic;color:#555;margin-left:auto; font-weight:normal;}
.move-info em, .move-info i {font-style:normal;font-weight:bold;color:#EA4335}

.text-deleted, .text-inserted, .text-modified, .text-matched { padding: 2px; white-space: pre-wrap; word-wrap: break-word; }
.text-deleted { background:#FFEDED; color:#721C24; }
.text-inserted { background:#EDF5FF; color:#004085; }
.text-modified { /* Base style for modified block */ } 
.text-matched { background:#D4EDDA; color:#155724; }

/* Word-level diffs */
.diff-word-deleted { background-color: #ffd1d1;  color: #721C24;font-weight: bold; }
.diff-word-inserted { background-color: #d1e7ff; color: #004085; font-weight: bold; }


.table-diff-container{display:flex;width:100%;}
.table-diff-side{flex:1;padding:5px;overflow-x:auto;}
.diff-table{width:100%;border-collapse:collapse;font-size:12px;table-layout:auto;}
.diff-table th,.diff-table td{border:1px solid #ddd;padding:6px;text-align:left;word-wrap:break-word;}
.diff-table th{background:#f2f2f2;font-weight:700;}
.cell-similar{background:#EDFFF0!important;border:1px solid #00AA00!important}
.cell-modified { /* Base for modified cells, word diffs will apply inside */ }
.cell-deleted{background:#FFEDED!important;border:1px solid #FF0000!important}
.cell-inserted{background:#EDF5FF!important;border:1px solid #0000FF!important}
.cell-empty { background: #ffffff; }

.metadata-table{width:100%;border-collapse:collapse;margin-bottom:15px}
.metadata-table th,.metadata-table td{border:1px solid #ddd;padding:8px;text-align:left}
.metadata-table th{background:#f2f2f2;width:200px}
@media(max-width:768px){
.difference-content,.table-diff-container{flex-direction:column}
.difference-side,.table-diff-side{border-right:none;border-bottom:1px solid #eee;margin-bottom:10px; min-width: 100%;}
.difference-side:last-child,.table-diff-side:last-child{border-bottom:none;margin-bottom:0}
}
</style>
<script>
document.addEventListener('DOMContentLoaded', function () {
    const pageButtons = document.querySelectorAll('.page-button');
    function setActivePage(hash) {
        pageButtons.forEach(btn => {
            btn.classList.remove('active');
            if (btn.getAttribute('href') === hash) {
                btn.classList.add('active');
            }
        });
        try {
            const targetSection = document.querySelector(hash);
            // if (targetSection) { targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        } catch (e) { console.warn("Could not scroll to section: " + hash, e); }
    }
    if (window.location.hash) {
        setActivePage(window.location.hash);
    } else if (pageButtons.length > 0) {
        const firstPageHash = pageButtons[0].getAttribute('href');
        setActivePage(firstPageHash);
    }
    window.addEventListener('hashchange', function () { setActivePage(window.location.hash); });
});
</script>
</head>
<body>
<div class="container">
<header>
  <h1>PDF Comparison: {{ pdf1_name }} vs {{ pdf2_name }}</h1>
  <p>Comparison Date: {{ now }}</p>
</header>

<table class="metadata-table">
  <tr><th>First PDF</th><td>{{ pdf1_name }}</td></tr>
  <tr><th>Second PDF</th><td>{{ pdf2_name }}</td></tr>
  {% for k,v in meta.items() %}
  <tr><th>{{ k }}</th><td>{{ escape_html(v) }}</td></tr>
  {% endfor %}
</table>

<div class="summary">
  <div class="stat-item"><div class="stat-label">Total Pages</div><div class="stat-value">{{ summary.total_pages }}</div></div>
  <div class="stat-item"><div class="stat-label">Pages with Differences</div><div class="stat-value">{{ summary.pages_with_differences }}</div></div>
  <div class="stat-item"><div class="stat-label">Text Differences</div><div class="stat-value">{{ summary.text_differences }}</div></div>
  <div class="stat-item"><div class="stat-label">Table Differences</div><div class="stat-value">{{ summary.table_differences }}</div></div>
</div>

<div class="diff-legend">
  <div class="legend-item"><div class="legend-color" style="background:#FFEDED;border:1px solid #FF0000"></div><span>Deleted (PDF1 only)</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#EDF5FF;border:1px solid #0000FF"></div><span>Inserted (PDF2 only)</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#FCF8E3;border:1px solid #EA4335"></div><span>Modified</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#D4EDDA;border:1px solid #155724"></div><span>Similar / Matched (Score  100%)</span></div>
  <div class="legend-item"><div class="legend-color" style="background:#D1ECF1;border:1px solid #0C5460"></div><span>Moved</span></div>
</div>

<div class="page-nav">
  {% for pg_num in pages.keys()|sort %}
    <a href="#page-{{ pg_num }}" class="page-button">{{ pg_num }}</a>
  {% endfor %}
</div>

{% for pg_num, page_content in pages.items()|sort %}
<section id="page-{{ pg_num }}" class="page-section">
  <div class="page-header">
    <h2 class="page-title">Page {{ pg_num }}</h2>
    {% if page_content.text_differences or page_content.table_differences %}
      <span class="status-badge status-modified">Has Differences</span>
    {% else %}
      <span class="status-badge status-matched">No Differences</span>
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
</section>
{% endfor %}
</div>
</body>
</html>
"""
