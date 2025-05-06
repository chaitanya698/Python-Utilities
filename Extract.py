"""
Extract.py 
----------
Enhanced PDF structure-preserving extractor with rule-based table detection.

Key improvements:
* Better detection of tables with missing borders using geometric analysis
* Enhanced multi-page table merging with rule-based continuation detection
* Improved text extraction under missing borders
* Better merging of continuation tables using string similarity
"""

import io
import logging
import uuid
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTPage, LTRect, LTLine
import pdfminer.high_level

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TableData:
    """
    Class for storing table data extracted from PDFs.
    This class is used as a container for table information.
    """
    table_id: str
    page_number: int
    bbox: Tuple[float, float, float, float]
    content: List[List[str]]
    is_continuation: bool = False
    continuation_of: Optional[str] = None

    def has_continuation_indicators(self):
        """Check if table has indicators suggesting it continues on next page."""
        if not self.content or len(self.content) == 0:
            return False
            
        # Check last row for continuation indicators
        last_row = self.content[-1]
        if not last_row:
            return False
            
        last_row_text = " ".join(cell.lower() for cell in last_row if cell)
        continuation_phrases = ["continued", "cont'd", "(continued)", "continues", "to be continued"]
        
        return any(phrase in last_row_text for phrase in continuation_phrases)
    
    def get_header_signature(self):
        """Get a signature of the header row for matching continuation tables."""
        if not self.content or len(self.content) < 1:
            return ""
            
        # Use the first row as header
        header = self.content[0]
        return "|".join(cell.strip().lower() for cell in header if cell.strip())


class PDFExtractor:
    """
    Enhanced PDF extractor with rule-based table detection and extraction.
    This class handles the extraction of content from PDF files.
    """
    
    def __init__(self, similarity_threshold=0.85, header_match_threshold=0.9, 
                 grid_detection_threshold=0.7, line_proximity_threshold=5.0):
        """Initialize the PDF extractor with customizable thresholds."""
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.grid_detection_threshold = grid_detection_threshold
        self.line_proximity_threshold = line_proximity_threshold
        self.continuation_tables = []
        
        logger.info("PDFExtractor initialized with the following thresholds:")
        logger.info(f"  similarity_threshold: {similarity_threshold}")
        logger.info(f"  header_match_threshold: {header_match_threshold}")
        logger.info(f"  grid_detection_threshold: {grid_detection_threshold}")
        logger.info(f"  line_proximity_threshold: {line_proximity_threshold}")
    
    def extract_pdf_content(self, pdf_content):
        """
        Extract structured content from PDF bytes.
        Returns a structure compatible with the compare.py expectations.
        """
        logger.info("Extracting content from PDF")
        
        # Create an in-memory file object
        pdf_file = io.BytesIO(pdf_content)
        
        # Data structure to store content
        pdf_data = {}
        
        # Extract pages using pdfminer
        pages = list(extract_pages(pdf_file, laparams=LAParams()))
        
        # Process each page
        for page_idx, page in enumerate(pages):
            page_num = page_idx + 1  # 1-based page numbering
            
            # Collect lines and rectangles for table border detection
            lines, rectangles = self._collect_lines_and_rectangles(page)
            
            # Extract text elements
            text_elements = self._extract_text_elements(page)
            
            # Extract tables using our rule-based table detection
            tables = self._detect_tables(page, text_elements, lines, rectangles)
            
            # Combine all elements
            elements = text_elements + tables
            
            # Extract full text
            page_text = self._extract_page_text(page)
            
            # Store page data
            pdf_data[page_num] = {
                "elements": elements,
                "page_number": page_num,
                "width": page.width,
                "height": page.height,
                "text": page_text
            }
        
        # Post-process to find continuation tables across pages
        pdf_data = self._process_continuation_tables(pdf_data)
        
        logger.info(f"Extracted {len(pdf_data)} pages with content")
        return pdf_data
    
    def _collect_lines_and_rectangles(self, page):
        """Collect lines and rectangles from the page for table border detection."""
        lines = []
        rectangles = []
        
        for element in page:
            if isinstance(element, LTRect):
                rectangles.append((element.x0, element.y0, element.x1, element.y1))
            elif isinstance(element, LTLine):
                lines.append((element.x0, element.y0, element.x1, element.y1))
                
        return lines, rectangles
    
    def _extract_text_elements(self, page):
        """Extract text elements from a page."""
        text_elements = []
        
        for element in page:
            if isinstance(element, LTTextBox):
                text = element.get_text().strip()
                if text:
                    bbox = (element.x0, element.y0, element.x1, element.y1)
                    text_elements.append({
                        "type": "text",
                        "content": text,
                        "bbox": bbox
                    })
        
        return text_elements
    
    def _extract_page_text(self, page):
        """Extract all text from a page as a single string."""
        texts = []
        for element in page:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        return "\n".join(texts)
    
    def _detect_tables(self, page, text_elements, lines, rectangles):
        """
        Rule-based table detection algorithm.
        Detects tables even with missing borders.
        """
        tables = []
        
        # First, try to detect tables using explicit borders
        border_tables = self._detect_tables_with_borders(page, text_elements, lines, rectangles)
        tables.extend(border_tables)
        
        # Then, detect tables using text alignment and layout
        text_alignment_tables = self._detect_tables_from_text_alignment(page, text_elements)
        
        # Filter out duplicate tables (those that significantly overlap with border tables)
        non_duplicate_tables = self._filter_duplicate_tables(text_alignment_tables, border_tables)
        tables.extend(non_duplicate_tables)
        
        # Process each detected table
        for table_idx, raw_table in enumerate(tables):
            # Generate unique ID for the table
            table_id = f"table_{page.pageid}_{uuid.uuid4().hex[:8]}"
            
            # Get content as list of lists (rows and cells)
            content = raw_table.get("content", [])
            
            # Calculate content hash for matching tables across pages
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Update table with additional info
            raw_table["table_id"] = table_id
            raw_table["content_hash"] = content_hash
            raw_table["has_nested_tables"] = False
            raw_table["nested_tables"] = []
            raw_table["type"] = "table"
            
        return tables
    
    def _detect_tables_with_borders(self, page, text_elements, lines, rectangles):
        """Detect tables with explicit borders using lines and rectangles."""
        tables = []
        
        # Identify potential table areas based on rectangles or grid of lines
        table_areas = []
        
        # Check for complete rectangles which may represent tables
        for rect in rectangles:
            table_areas.append(rect)
        
        # Check for grids formed by lines
        if lines:
            # Find horizontal lines
            h_lines = [(x0, y0, x1, y1) for x0, y0, x1, y1 in lines if abs(y1 - y0) < self.line_proximity_threshold]
            
            # Find vertical lines
            v_lines = [(x0, y0, x1, y1) for x0, y0, x1, y1 in lines if abs(x1 - x0) < self.line_proximity_threshold]
            
            # If we have both horizontal and vertical lines, try to find grids
            if h_lines and v_lines:
                # Sort by y-coordinate
                h_lines.sort(key=lambda l: l[1])  # sort by y0
                
                # Sort by x-coordinate
                v_lines.sort(key=lambda l: l[0])  # sort by x0
                
                # Group horizontal lines that are close to each other
                h_line_groups = []
                current_group = [h_lines[0]]
                
                for i in range(1, len(h_lines)):
                    if abs(h_lines[i][1] - h_lines[i-1][1]) < self.line_proximity_threshold:
                        current_group.append(h_lines[i])
                    else:
                        if len(current_group) > 1:
                            h_line_groups.append(current_group)
                        current_group = [h_lines[i]]
                
                if len(current_group) > 1:
                    h_line_groups.append(current_group)
                
                # Group vertical lines that are close to each other
                v_line_groups = []
                current_group = [v_lines[0]]
                
                for i in range(1, len(v_lines)):
                    if abs(v_lines[i][0] - v_lines[i-1][0]) < self.line_proximity_threshold:
                        current_group.append(v_lines[i])
                    else:
                        if len(current_group) > 1:
                            v_line_groups.append(current_group)
                        current_group = [v_lines[i]]
                
                if len(current_group) > 1:
                    v_line_groups.append(current_group)
                
                # Identify grid areas from intersecting line groups
                for h_group in h_line_groups:
                    for v_group in v_line_groups:
                        # Calculate potential grid area
                        x0 = min(l[0] for l in v_group)
                        y0 = min(l[1] for l in h_group)
                        x1 = max(l[2] for l in v_group)
                        y1 = max(l[3] for l in h_group)
                        
                        # Check if the grid has sufficient cells
                        if len(h_group) >= 2 and len(v_group) >= 2:
                            table_areas.append((x0, y0, x1, y1))
        
        # For each potential table area, collect text elements inside
        for x0, y0, x1, y1 in table_areas:
            table_text_elements = []
            
            for element in text_elements:
                ex0, ey0, ex1, ey1 = element["bbox"]
                
                # Check if element is inside the table area
                if (ex0 >= x0 and ex1 <= x1 and ey0 >= y0 and ey1 <= y1):
                    table_text_elements.append(element)
            
            # If we have text elements, try to organize them into a table
            if table_text_elements:
                # Sort by y-coordinate (top to bottom)
                table_text_elements.sort(key=lambda e: e["bbox"][1], reverse=True)
                
                # Group by similar y-coordinates (rows)
                rows = []
                current_row = [table_text_elements[0]]
                current_y = table_text_elements[0]["bbox"][1]
                
                for i in range(1, len(table_text_elements)):
                    element = table_text_elements[i]
                    if abs(element["bbox"][1] - current_y) < self.line_proximity_threshold:
                        current_row.append(element)
                    else:
                        # Sort current row by x-coordinate
                        current_row.sort(key=lambda e: e["bbox"][0])
                        
                        # Extract text
                        row_content = [element["content"] for element in current_row]
                        rows.append(row_content)
                        
                        # Start new row
                        current_row = [element]
                        current_y = element["bbox"][1]
                
                # Add the last row
                if current_row:
                    current_row.sort(key=lambda e: e["bbox"][0])
                    row_content = [element["content"] for element in current_row]
                    rows.append(row_content)
                
                # If we have a reasonable table structure, add it
                if len(rows) >= 2 and all(len(row) > 1 for row in rows):
                    tables.append({
                        "content": rows,
                        "bbox": (x0, y0, x1, y1)
                    })
        
        return tables
    
    def _detect_tables_from_text_alignment(self, page, text_elements):
        """Detect tables using text alignment and layout patterns."""
        # Group text elements into potential table rows based on vertical alignment
        rows = self._group_elements_into_rows(text_elements)
        
        # Identify potential tables based on row patterns
        table_candidates = []
        
        # Check for sequences of rows with similar structure (column count)
        i = 0
        while i < len(rows):
            # Need at least 2 rows to form a table
            if i + 1 >= len(rows):
                i += 1
                continue
            
            current_row = rows[i]
            next_row = rows[i + 1]
            
            # Check if these rows might form a table (similar column count)
            if abs(len(current_row) - len(next_row)) <= 1:
                # Start a potential table with these rows
                potential_table_rows = [current_row, next_row]
                row_index = i + 2
                
                # Add subsequent rows with similar structure
                while row_index < len(rows):
                    next_row = rows[row_index]
                    # Check if the row has a similar structure to the first row
                    if abs(len(next_row) - len(current_row)) <= 1:
                        potential_table_rows.append(next_row)
                        row_index += 1
                    else:
                        break
                
                # If we have at least 2 rows with similar structure, consider it a table
                if len(potential_table_rows) >= 2:
                    # Calculate bounding box
                    all_elements = [elem for row in potential_table_rows for elem in row]
                    x0 = min(elem["bbox"][0] for elem in all_elements)
                    y0 = min(elem["bbox"][1] for elem in all_elements)
                    x1 = max(elem["bbox"][2] for elem in all_elements)
                    y1 = max(elem["bbox"][3] for elem in all_elements)
                    
                    # Extract table content
                    content = []
                    for row in potential_table_rows:
                        row_content = [elem["content"] for elem in row]
                        content.append(row_content)
                    
                    table_candidates.append({
                        "content": content,
                        "bbox": (x0, y0, x1, y1)
                    })
                    
                    i = row_index  # Skip the rows we've processed
                else:
                    i += 1
            else:
                i += 1
        
        # Check for sequences of rows with consistent alignment (column x-positions)
        tables_from_alignment = self._identify_tables_from_column_alignment(text_elements)
        table_candidates.extend(tables_from_alignment)
        
        # Remove duplicate tables
        return self._filter_duplicate_tables(table_candidates, [])
    
    def _group_elements_into_rows(self, text_elements):
        """Group text elements into rows based on vertical position."""
        # Sort by y-coordinate (top to bottom)
        sorted_elements = sorted(text_elements, key=lambda e: -e["bbox"][1])  # Negative for top-to-bottom
        
        rows = []
        if not sorted_elements:
            return rows
            
        current_row = [sorted_elements[0]]
        current_y = sorted_elements[0]["bbox"][1]
        
        for i in range(1, len(sorted_elements)):
            element = sorted_elements[i]
            
            # Check if this element is on the same row
            if abs(element["bbox"][1] - current_y) < self.line_proximity_threshold:
                current_row.append(element)
            else:
                # Sort current row by x-coordinate (left to right)
                current_row.sort(key=lambda e: e["bbox"][0])
                rows.append(current_row)
                
                # Start new row
                current_row = [element]
                current_y = element["bbox"][1]
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda e: e["bbox"][0])
            rows.append(current_row)
        
        return rows
    
    def _identify_tables_from_column_alignment(self, text_elements):
        """Identify tables based on consistent column alignment."""
        rows = self._group_elements_into_rows(text_elements)
        
        # No rows or too few rows
        if len(rows) < 2:
            return []
        
        # Find sequences of rows with consistent column x-positions
        tables = []
        i = 0
        
        while i < len(rows) - 1:
            # Check column alignment between this row and the next
            current_row = rows[i]
            
            # Skip rows with only one element (not likely to be table rows)
            if len(current_row) <= 1:
                i += 1
                continue
            
            # Start with current row as the first row of a potential table
            potential_table_rows = [current_row]
            column_positions = [elem["bbox"][0] for elem in current_row]
            
            # Check subsequent rows for similar column alignment
            row_index = i + 1
            while row_index < len(rows):
                next_row = rows[row_index]
                
                # Skip rows with only one element
                if len(next_row) <= 1:
                    row_index += 1
                    continue
                
                next_positions = [elem["bbox"][0] for elem in next_row]
                
                # Check if column positions are similar
                if self._check_column_alignment(column_positions, next_positions):
                    potential_table_rows.append(next_row)
                    row_index += 1
                else:
                    break
            
            # If we have at least 3 rows with consistent column alignment, consider it a table
            if len(potential_table_rows) >= 3:
                # Calculate bounding box
                all_elements = [elem for row in potential_table_rows for elem in row]
                x0 = min(elem["bbox"][0] for elem in all_elements)
                y0 = min(elem["bbox"][1] for elem in all_elements)
                x1 = max(elem["bbox"][2] for elem in all_elements)
                y1 = max(elem["bbox"][3] for elem in all_elements)
                
                # Extract table content
                content = []
                for row in potential_table_rows:
                    row_content = [elem["content"] for elem in row]
                    content.append(row_content)
                
                tables.append({
                    "content": content,
                    "bbox": (x0, y0, x1, y1)
                })
                
                i = row_index  # Skip the rows we've processed
            else:
                i += 1
        
        return tables
    
    def _check_column_alignment(self, positions1, positions2):
        """Check if two sets of column positions are aligned."""
        # Different number of columns - allow for off-by-one
        if abs(len(positions1) - len(positions2)) > 1:
            return False
        
        # Sort positions
        pos1 = sorted(positions1)
        pos2 = sorted(positions2)
        
        # Check alignment of available columns
        min_cols = min(len(pos1), len(pos2))
        aligned_cols = 0
        
        for i in range(min_cols):
            if abs(pos1[i] - pos2[i]) < self.line_proximity_threshold:
                aligned_cols += 1
        
        # Consider aligned if a high percentage of columns are aligned
        alignment_ratio = aligned_cols / min_cols
        return alignment_ratio >= self.grid_detection_threshold
    
    def _filter_duplicate_tables(self, candidates, existing_tables):
        """Filter out duplicate table candidates."""
        if not candidates:
            return []
            
        # If no existing tables, just check against each other
        if not existing_tables:
            filtered = [candidates[0]]
            
            for candidate in candidates[1:]:
                is_duplicate = False
                
                for existing in filtered:
                    if self._is_same_table(candidate, existing):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered.append(candidate)
            
            return filtered
        
        # Check against existing tables
        filtered = []
        
        for candidate in candidates:
            is_duplicate = False
            
            for existing in existing_tables:
                if self._is_same_table(candidate, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(candidate)
        
        return filtered
    
    def _is_same_table(self, table1, table2):
        """Check if two table candidates represent the same table."""
        # Get bounding boxes
        bbox1 = table1.get("bbox", (0, 0, 0, 0))
        bbox2 = table2.get("bbox", (0, 0, 0, 0))
        
        # Calculate overlap ratio
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        if area1 <= 0 or area2 <= 0:
            return False
            
        overlap_area = x_overlap * y_overlap
        overlap_ratio1 = overlap_area / area1
        overlap_ratio2 = overlap_area / area2
        
        # If one table largely overlaps the other, consider them the same
        return overlap_ratio1 > 0.7 or overlap_ratio2 > 0.7
    
    def _process_continuation_tables(self, pdf_data):
        """
        Identify and merge tables that continue across multiple pages.
        """
        # Collect all tables from all pages
        all_tables = []
        for page_num, page_data in pdf_data.items():
            for element in page_data.get("elements", []):
                if element.get("type") == "table":
                    # Add page number to the table data
                    table_with_page = element.copy()
                    table_with_page["page_num"] = page_num
                    all_tables.append(table_with_page)
        
        # Sort tables by page number
        all_tables.sort(key=lambda t: t["page_num"])
        
        # Find continuation tables
        merged_tables = {}  # Map of merged table IDs to list of component table IDs
        
        for i, table in enumerate(all_tables):
            # Skip tables already identified as continuations
            if table.get("table_id") in merged_tables:
                continue
                
            # Check subsequent tables for potential continuations
            for j in range(i + 1, len(all_tables)):
                next_table = all_tables[j]
                
                # Skip if not on the next page
                if next_table["page_num"] != table["page_num"] + 1:
                    continue
                
                # Check if headers match (indicating continuation)
                if self._check_table_continuation(table, next_table):
                    # Mark as continuation
                    if table["table_id"] not in merged_tables:
                        merged_tables[table["table_id"]] = [table["table_id"]]
                    
                    merged_tables[table["table_id"]].append(next_table["table_id"])
                    
                    # Update the next table to indicate it's a continuation
                    next_table["is_continuation"] = True
                    next_table["continuation_of"] = table["table_id"]
                    
                    # Update the table content in the original data structure
                    pdf_data[next_table["page_num"]]["elements"] = [
                        elem if elem.get("table_id") != next_table["table_id"] else next_table
                        for elem in pdf_data[next_table["page_num"]]["elements"]
                    ]
                    
                    break  # Only find the first continuation
        
        # Now merge content for continuation tables
        for primary_id, table_ids in merged_tables.items():
            if len(table_ids) <= 1:
                continue
                
            # Find the primary table
            primary_table = None
            for page_data in pdf_data.values():
                for element in page_data.get("elements", []):
                    if element.get("type") == "table" and element.get("table_id") == primary_id:
                        primary_table = element
                        break
                if primary_table:
                    break
            
            if not primary_table:
                continue
                
            # Collect all continuation tables
            continuation_tables = []
            for table_id in table_ids[1:]:  # Skip the primary table
                for page_data in pdf_data.values():
                    for element in page_data.get("elements", []):
                        if element.get("type") == "table" and element.get("table_id") == table_id:
                            continuation_tables.append(element)
                            break
            
            # Merge the content, skipping header rows in continuation tables
            merged_content = primary_table["content"].copy()
            for cont_table in continuation_tables:
                # Skip the header row (first row) assuming it's a repeat
                if len(cont_table["content"]) > 1:
                    merged_content.extend(cont_table["content"][1:])
            
            # Update the primary table with merged content
            primary_table["content"] = merged_content
            primary_table["has_continuation"] = True
            primary_table["continuation_ids"] = table_ids[1:]
            
            # Calculate new content hash
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in merged_content)
            primary_table["content_hash"] = hashlib.md5(content_str.encode()).hexdigest()
        
        return pdf_data
    
    def _check_table_continuation(self, table1, table2):
        """
        Check if table2 is a continuation of table1 by comparing headers
        and looking for continuation indicators.
        """
        # Get header rows
        if not table1.get("content") or not table2.get("content"):
            return False
            
        header1 = table1["content"][0] if table1["content"] else []
        header2 = table2["content"][0] if table2["content"] else []
        
        # Check for header similarity
        header_similarity = self._calculate_header_similarity(header1, header2)
        
        # Check if table1 has continuation indicators
        last_row = table1["content"][-1] if table1["content"] else []
        last_row_text = " ".join(str(cell).lower() for cell in last_row)
        has_continuation_indicator = any(phrase in last_row_text 
                                       for phrase in ["continued", "cont'd", "(continued)", 
                                                     "continues", "to be continued"])
        
        # Consider it a continuation if headers are similar or there's a continuation indicator
        return header_similarity >= self.header_match_threshold or has_continuation_indicator
    
    def _calculate_header_similarity(self, header1, header2):
        """Calculate similarity between two table headers using simple string matching."""
        if not header1 or not header2:
            return 0.0
        
        # Simple similarity: percentage of matching cell values
        matches = 0
        total = max(len(header1), len(header2))
        
        for i in range(min(len(header1), len(header2))):
            # Normalize and compare cell text
            cell1 = str(header1[i]).strip().lower()
            cell2 = str(header2[i]).strip().lower()
            
            if cell1 == cell2:
                matches += 1
            elif cell1 in cell2 or cell2 in cell1:
                matches += 0.7  # Partial match
            # Add Levenshtein distance for fuzzy matching
            elif self._levenshtein_similarity(cell1, cell2) > 0.7:
                matches += 0.5  # Fuzzy match
        
        return matches / total if total > 0 else 0.0

    def _levenshtein_similarity(self, s1, s2):
        """Calculate similarity based on Levenshtein distance."""
        # Calculate Levenshtein distance
        if len(s1) < len(s2):
            return self._levenshtein_similarity(s2, s1)
        
        if len(s2) == 0:
            return 0.0
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        # Convert distance to similarity (0 to 1)
        max_len = max(len(s1), len(s2))
        distance = previous_row[-1]
        similarity = 1 - (distance / max_len) if max_len > 0 else 0
        
        return similarity
