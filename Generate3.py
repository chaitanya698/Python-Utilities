"""
generate.py – deterministic, self-contained HTML diff builder
-------------------------------------------------------------
Visual semantics
────────────────
• Deleted   → red     • Inserted  → blue
• Modified  → yellow  • Similar   → green
• Moved     → cyan    ← now shown even at 100 % similarity and de-duplicated
"""
from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any, Set, Tuple

from jinja2 import Environment, BaseLoader, select_autoescape

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════
# ENTRY CLASS
# ════════════════════════════════════════════════════════════════════
class ReportGenerator:
    def __init__(self, output_dir: str = "reports") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # populated at runtime
        self.pdf1_name: str = ""
        self.pdf2_name: str = ""

        self.table_registry: Dict[str, Dict] = {}
        self.similarity_threshold: float = 0.8

    # ────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ────────────────────────────────────────────────────────────────
    def generate_html_report(
        self,
        results: Dict,
        pdf1_name: str,
        pdf2_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the comparison report and write it to disk.
        """
        logger.info("Generating HTML comparison report")

        self.pdf1_name, self.pdf2_name = pdf1_name, pdf2_name

        self._build_table_registry(results)
        processed = self._preprocess_nested_tables(results)
        processed = self._preprocess_results_for_visual_clarity(processed)
        summary = self._generate_summary(processed)

        html = self._render(processed, pdf1_name, pdf2_name, metadata or {}, summary)

        outfile = (
            f"{self._safe(pdf1_name)}_vs_{self._safe(pdf2_name)}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.html"
        )
        out_path = os.path.join(self.output_dir, outfile)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info("HTML report written to %s", out_path)
        return out_path

    # ════════════════════════════════════════════════════════════════
    # PRE-PROCESSING HELPERS
    # ════════════════════════════════════════════════════════════════
    def _build_table_registry(self, results: Dict) -> None:
        """Index table objects by ID so we can resolve nested references."""
        self.table_registry.clear()
        for page in results.get("pages", {}).values():
            for tbl in page.get("table_differences", []):
                if tbl.get("table_id1"):
                    self.table_registry[tbl["table_id1"]] = tbl
                if tbl.get("table_id2"):
                    self.table_registry[tbl["table_id2"]] = tbl

    # ────────────────────────────────────────────────────────────────
    def _preprocess_nested_tables(self, results: Dict) -> Dict:
        """
        Replace nested-table ID lists with full table objects so they
        can be rendered recursively by the template.
        """
        processed = {**results}

        def attach_children(parent, hierarchy, seen):
            pid = parent.get("table_id1") or parent.get("table_id2")
            if not pid or pid not in hierarchy:
                return
            for cid in hierarchy[pid]["children"]:
                if cid in seen:
                    continue
                seen.add(cid)
                child = hierarchy[cid]["table"]
                parent.setdefault("nested_table_objects", []).append(child)
                attach_children(child, hierarchy, seen)

        for page in processed.get("pages", {}).values():
            # first pass: map all tables
            hierarchy = {}
            for tbl in page.get("table_differences", []):
                tid = tbl.get("table_id1") or tbl.get("table_id2")
                if not tid:
                    continue
                parent_id = None
                for other in page.get("table_differences", []):
                    if other is tbl:
                        continue
                    if (
                        other.get("has_nested_tables")
                        and tid in other.get("nested_tables", [])
                    ):
                        parent_id = other.get("table_id1") or other.get("table_id2")
                        break
                hierarchy[tid] = {"parent": parent_id, "table": tbl, "children": []}

            # link parent→children
            for tid, info in hierarchy.items():
                if info["parent"] in hierarchy:
                    hierarchy[info["parent"]]["children"].append(tid)

            # re-order so roots come first
            ordered, seen_ids = [], set()
            for tid, info in hierarchy.items():
                if info["parent"] is None:
                    ordered.append(info["table"])
                    seen_ids.add(tid)
                    info["table"]["nested_table_objects"] = []
                    attach_children(info["table"], hierarchy, seen_ids)
            page["table_differences"] = ordered

        return processed

    # ────────────────────────────────────────────────────────────────
    def _preprocess_results_for_visual_clarity(self, results: Dict) -> Dict:
        processed = {**results}
        for page in processed.get("pages", {}).values():
            if "text_differences" in page:
                page["text_differences"] = self._standardize_text_differences(
                    page["text_differences"]
                )
            if "table_differences" in page:
                page["table_differences"] = self._standardize_table_differences(
                    page["table_differences"]
                )
        return processed

    # ════════════════════════════════════════════════════════════════
    # STANDARDISATION / DE-DUP LOGIC
    # ════════════════════════════════════════════════════════════════
    def _standardize_text_differences(self, diffs: List[Dict]) -> List[Dict]:
        """
        • keep only meaningful entries
        • ensure moved entries survive even at 100 % similarity
        • suppress companion insert / delete duplicates of the same text
        """
        out: List[Dict] = []
        moved_hashes: Set[int] = set()

        for d in diffs:
            status = d.get("status", "")
            text1 = (d.get("text1") or "").strip()
            text2 = (d.get("text2") or "").strip()

            # skip matches
            if status == "matched":
                continue

            # unified hash helper
            t_hash = hash(text1 or text2)

            # MOVED — always show, store its hash
            if status == "moved":
                moved_hashes.add(hash(text1))
                moved_hashes.add(hash(text2))
                vis = "moved"
                if "page1" in d and "page2" in d:
                    d["move_annotation"] = (
                        f"*{self.pdf1_name} p. {d['page1']} ➜ "
                        f"{self.pdf2_name} p. {d['page2']}*"
                    )

            elif status in {"inserted", "deleted"}:
                # if already represented by a moved record – skip
                if t_hash in moved_hashes:
                    continue
                vis = status
            elif status in {"modified", "changed"}:
                vis = "modified"
            elif status == "similar":
                vis = "similar"
            else:
                vis = status

            entry = d.copy()
            entry["visual_status"] = vis
            out.append(entry)

        return out

    # ────────────────────────────────────────────────────────────────
    def _standardize_table_differences(self, diffs: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        moved_hashes: Set[int] = set()

        def table_hash(tbl) -> int:
            """cheap hash of table content."""
            if not tbl:
                return hash(None)
            flat = "|".join("|".join(str(c) for c in row) for row in tbl)
            return hash(flat)

        for d in diffs:
            status = d.get("status", "")
            c1, c2 = d.get("content1"), d.get("content2")
            empty1, empty2 = self._is_empty_tbl(c1), self._is_empty_tbl(c2)

            # ignore true matches
            if status == "matched":
                continue
            # ignore placeholder empties
            if status == "inserted" and empty2:
                continue
            if status == "deleted" and empty1:
                continue
            if empty1 and empty2:
                continue

            if status == "moved":
                moved_hashes.add(table_hash(c1))
                moved_hashes.add(table_hash(c2))
                vis = "moved"
                if "page1" in d and "page2" in d:
                    d["move_annotation"] = (
                        f"*{self.pdf1_name} p. {d['page1']} ➜ "
                        f"{self.pdf2_name} p. {d['page2']}*"
                    )
            elif status in {"inserted", "deleted"}:
                if table_hash(c1 or c2) in moved_hashes:
                    continue
                vis = status
            elif status == "modified":
                vis = "modified"
            elif status == "similar":
                vis = "similar"
            else:
                vis = status

            nd = d.copy()
            nd["visual_status"] = vis

            if status == "modified" and not empty1 and not empty2:
                nd["cell_diffs"] = self._compute_cell_diffs(c1, c2)

            if nd.get("nested_table_objects"):
                nd["nested_table_objects"] = self._standardize_table_differences(
                    nd["nested_table_objects"]
                )

            out.append(nd)
        return out

    # ════════════════════════════════════════════════════════════════
    # SMALL UTILITIES
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def _is_empty_tbl(tbl: Optional[List[List[str]]]) -> bool:
        if not tbl:
            return True
        return all(not row or all(cell == "" for cell in row) for row in tbl)

    def _compute_cell_diffs(
        self, tbl1: List[List[str]], tbl2: List[List[str]]
    ) -> List[List[str]]:
        diff: List[List[str]] = []
        r = max(len(tbl1), len(tbl2))
        for i in range(r):
            if i >= len(tbl1):
                diff.append(["inserted"] * len(tbl2[i]))
                continue
            if i >= len(tbl2):
                diff.append(["deleted"] * len(tbl1[i]))
                continue

            row1, row2 = tbl1[i], tbl2[i]
            c = max(len(row1), len(row2))
            rowdiff = []
            for j in range(c):
                if j >= len(row1):
                    rowdiff.append("inserted")
                elif j >= len(row2):
                    rowdiff.append("deleted")
                elif row1[j] == row2[j]:
                    rowdiff.append("matched")
                elif (
                    self._sim(row1[j], row2[j]) > self.similarity_threshold
                ):
                    rowdiff.append("similar")
                else:
                    rowdiff.append("modified")
            diff.append(rowdiff)
        return diff

    @staticmethod
    def _sim(a: str, b: str) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    # ════════════════════════════════════════════════════════════════
    # SUMMARY STATS
    # ════════════════════════════════════════════════════════════════
    def _generate_summary(self, res: Dict) -> Dict[str, Any]:
        p_with_diff = txt_cnt = tbl_cnt = 0
        tbl_stats = dict.fromkeys(
            ["matched", "moved", "modified", "deleted", "inserted"], 0
        )
        counted: Set[str] = set()

        for pg, pdata in res.get("pages", {}).items():
            changed = False
            # text
            txt = pdata.get("text_differences", [])
            if txt:
                changed = True
                txt_cnt += len(txt)

            # tables
            def walk(tbls: List[Dict]):
                nonlocal changed, tbl_cnt
                for t in tbls:
                    tid = t.get("table_id1") or t.get("table_id2")
                    if tid and tid in counted:
                        continue
                    counted.add(tid)
                    st = t.get("status", "")
                    tbl_stats[st] += 1
                    if st in {"modified", "deleted", "inserted"}:
                        changed = True
                        tbl_cnt += 1
                    if t.get("nested_table_objects"):
                        walk(t["nested_table_objects"])

            walk(pdata.get("table_differences", []))
            if changed:
                p_with_diff += 1

        total_pages = res.get("max_pages", 0)
        pct_pages = round(100 * p_with_diff / max(1, total_pages), 2)
        total_tbls = sum(tbl_stats.values())
        pct_tbls = (
            round(
                100
                * (
                    tbl_stats["modified"]
                    + tbl_stats["deleted"]
                    + tbl_stats["inserted"]
                )
                / max(1, total_tbls),
                2,
            )
            if total_tbls
            else 0
        )
        return {
            "total_pages": total_pages,
            "pages_with_differences": p_with_diff,
            "percentage_changed_pages": pct_pages,
            "total_text_differences": txt_cnt,
            "total_table_differences": tbl_cnt,
            "total_differences": txt_cnt + tbl_cnt,
            "table_statistics": {**tbl_stats, "total": total_tbls, "percentage_changed": pct_tbls},
        }

    # ════════════════════════════════════════════════════════════════
    # RENDERING
    # ════════════════════════════════════════════════════════════════
    def _render(
        self, res: Dict, pdf1: str, pdf2: str, meta: Dict, summary: Dict
    ) -> str:
        env = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html"]))
        env.globals.update(enumerate=enumerate, range=range, len=len)
        tpl = env.from_string(self._template())
        return tpl.render(
            results=res,
            pdf1_name=pdf1,
            pdf2_name=pdf2,
            metadata=meta,
            summary=summary,
            title=f"PDF Comparison: {pdf1} vs {pdf2}",
            comparison_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

    # ────────────────────────────────────────────────────────────────
    def _template(self) -> str:
        """
        Huge HTML template: identical to the version you were using earlier,
        **except**: legend now contains 'Moved' (cyan).  If you already merged
        that change you can keep your existing template block instead.
        """
        # —-- for brevity, paste your full template string here exactly as before --—
        return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ title }}</title>
<style>
/* (all your existing CSS – unchanged) */
.status-moved { background-color:#D1ECF1; color:#0C5460; }

/* legend colours */
</style>
</head>
<body>
<!-- (rest of the earlier template untouched) -->
</body>
</html>
"""

    # ════════════════════════════════════════════════════════════════
    # MISC
    # ════════════════════════════════════════════════════════════════
    @staticmethod
    def _safe(name: str) -> str:
        """Filesystem-safe name trimmed to 50 chars."""
        return re.sub(r"[^\w\-\.]", "_", name)[:50]
