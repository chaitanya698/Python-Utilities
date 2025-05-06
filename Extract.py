"""
extract.py
----------
Deterministic PDF content extraction with advanced table detection.

Key improvements:
* Better detection of tables with missing borders using positional analysis
* Enhanced multi-page table merging with continuation header detection
* Improved text extraction under missing borders
* Better merging of continuation tables
"""

import io
import logging
import uuid
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import math

# PDF processing libraries
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTPage
from pdfminer.layout import LTRect, LTLine
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

    def has_continuation_indicators(self) -> bool:
        """Check if table has indicators suggesting it continues on next page."""
        if not self.content or len(self.content) == 0:
            return False
            
        # Check last row for continuation indicators
        last_row = self.content[-1]
        if not last_row:
            return False
            
        last_row_text = " ".join(cell.lower() for cell in last_row if cell)
        continuation_phrases = ["continued", "cont'd", "(continued)", "continues", 
                               "to be continued", "(cont)", "(contd)"]
        
        return any(phrase in last_row_text for phrase in continuation_phrases)
    
    def get_header_signature(self) -> str:
        """Get a signature of the header row for matching continuation tables."""
        if not self.content or len(self.content) < 1:
            return ""
            
        # Use the first row as header
        header = self.content[0]
        return "|".join(cell.strip().lower() for cell in header if cell.strip())


class PDFExtractor:
    """
    Enhanced PDF extractor with improved table detection and extraction.
    Uses deterministic methods for content extraction with no ML.
    """
    
    def __init__(self, 
                similarity_threshold: float = 0.85, 
                header_match_threshold: float = 0.9, 
                nested_table_threshold: float = 0.85, 
                nested_area_ratio: float = 0.75):
        """Initialize the PDF extractor with customizable thresholds."""
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.nested_table_threshold = nested_table_threshold
        self.nested_area_ratio = nested_area_ratio
        self.continuation_tables = []
        
        logger.info("PDFExtractor initialized with the following thresholds:")
        logger.info(f"  similarity_threshold: {similarity_threshold}")
        logger.info(f"  header_match_threshold: {header_match_threshold}")
        logger.info(f"  nested_table_threshold: {nested_table_threshold}")
        logger.info(f"  nested_area_ratio: {nested_area_ratio}")
    
    def extract_pdf_content(self, pdf_content: bytes) -> Dict:
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
        laparams = LAParams(
            line_margin=0.5,   # Adjust line margin for better text block detection
            char_margin=2.0,   # Increase char margin to better handle spacing
            word_margin=0.1,   # Lower word margin to better group words
            boxes_flow=0.5,    # Adjust boxes flow for improved text flow
            detect_vertical=True  # Enable vertical text detection
        )
        pages = list(extract_pages(pdf_file, laparams=laparams))
        
        # Process each page
        for page_idx, page in enumerate(pages):
            page_num = page_idx + 1  # 1-based page numbering
            
            # Extract line elements for table border detection
            lines = self._extract_lines(page)
            
            # Extract text elements
            text_elements = self._extract_text_elements(page)
            
            # Extract tables using our enhanced table detection
            tables = self._detect_tables(page, text_elements, lines)
            
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
    
    def _extract_text_elements(self, page: LTPage) -> List[Dict]:
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
    
    def _extract_lines(self, page: LTPage) -> List[Tuple]:
        """Extract line elements that could be table borders."""
        lines = []
        
        for obj in page:
            # Extract horizontal and vertical lines
            if isinstance(obj, LTRect):
                # Rectangles (often table cells)
                x0, y0, x1, y1 = obj.bbox
                if x1 - x0 < 2:  # Vertical line
                    lines.append(('v', x0, y0, y1))
                elif y1 - y0 < 2:  # Horizontal line
                    lines.append(('h', y0, x0, x1))
                else:
                    # Add all 4 sides of the rectangle
                    lines.append(('h', y0, x0, x1))  # Bottom
                    lines.append(('h', y1, x0, x1))  # Top
                    lines.append(('v', x0, y0, y1))  # Left
                    lines.append(('v', x1, y0, y1))  # Right
            
            elif isinstance(obj, LTLine):
                # Simple lines
                x0, y0, x1, y1 = obj.bbox
                if abs(x1 - x0) < 2:  # Vertical line
                    lines.append(('v', x0, min(y0, y1), max(y0, y1)))
                elif abs(y1 - y0) < 2:  # Horizontal line
                    lines.append(('h', y0, min(x0, x1), max(x0, x1)))
        
        return lines
    
    def _extract_page_text(self, page: LTPage) -> str:
        """Extract all text from a page as a single string."""
        texts = []
        for element in page:
            if isinstance(element, LTTextContainer):
                texts.append(element.get_text())
        return "\n".join(texts)
    
    def _detect_tables(self, page: LTPage, text_elements: List[Dict], lines: List[Tuple]) -> List[Dict]:
        """
        Enhanced table detection algorithm using deterministic methods.
        Detects tables with or without explicit borders.
        """
        tables = []
        
        # Process tables based on explicit borders first
        border_tables = self._detect_tables_by_borders(page, text_elements, lines)
        
        # Process tables based on alignment patterns for borderless tables
        alignment_tables = self._detect_tables_by_alignment(page, text_elements, lines)
        
        # Combine and remove overlaps
        raw_tables = self._merge_table_candidates(border_tables + alignment_tables)
        
        # Process each detected table
        for table_idx, raw_table in enumerate(raw_tables):
            # Generate unique ID for the table
            table_id = f"table_{page.pageid}_{uuid.uuid4().hex[:8]}"
            
            # Get content as list of lists (rows and cells)
            content = raw_table.get("content", [])
            
            # Calculate content hash for matching tables across pages
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Check for nested tables
            nested_tables, has_nested = self._check_for_nested_tables(raw_table, raw_tables)
            
            # Create table element
            table_element = {
                "type": "table",
                "table_id": table_id,
                "content": content,
                "bbox": raw_table.get("bbox", (0, 0, 0, 0)),
                "content_hash": content_hash,
                "has_nested_tables": has_nested,
                "nested_tables": nested_tables
            }
            
            tables.append(table_element)
        
        return tables
    
    def _detect_tables_by_borders(self, page: LTPage, text_elements: List[Dict], lines: List[Tuple]) -> List[Dict]:
        """Detect tables using explicit border lines."""
        tables = []
        
        # First check if we have enough lines to form tables
        if len(lines) < 4:  # Need at least 4 lines to form a table
            return []
        
        # Group horizontal and vertical lines
        h_lines = sorted([l for l in lines if l[0] == 'h'], key=lambda x: x[1])
        v_lines = sorted([l for l in lines if l[0] == 'v'], key=lambda x: x[1])
        
        # If we don't have enough lines, return empty
        if len(h_lines) < 2 or len(v_lines) < 2:
            return []
        
        # Find potential table boundaries by looking for rectangular areas
        potential_tables = []
        for i, h1 in enumerate(h_lines[:-1]):
            for j, h2 in enumerate(h_lines[i+1:], i+1):
                # Check if two horizontal lines could form top and bottom of a table
                top_y = max(h1[1], h2[1])
                bottom_y = min(h1[1], h2[1])
                if top_y - bottom_y < 20:  # Too narrow to be a table
                    continue
                
                # Find vertical lines that could be left and right boundaries
                matching_v_lines = []
                for v in v_lines:
                    v_x, v_ymin, v_ymax = v[1], v[2], v[3]
                    # Check if vertical line spans between the horizontal lines
                    if v_ymin <= bottom_y + 5 and v_ymax >= top_y - 5:
                        matching_v_lines.append(v)
                
                # We need at least 2 vertical lines for a table
                if len(matching_v_lines) < 2:
                    continue
                
                # Sort vertical lines by x position
                matching_v_lines.sort(key=lambda x: x[1])
                
                # Check for text elements inside the table boundaries
                for left_idx, left_v in enumerate(matching_v_lines[:-1]):
                    for right_idx, right_v in enumerate(matching_v_lines[left_idx+1:], left_idx+1):
                        # Potential table area
                        left_x = left_v[1]
                        right_x = right_v[1]
                        
                        # Check if this area contains text elements
                        contained_texts = []
                        for elem in text_elements:
                            x0, y0, x1, y1 = elem["bbox"]
                            text = elem["content"]
                            
                            # Check if text is inside the potential table
                            if (left_x - 5 <= x0 <= right_x + 5 and 
                                bottom_y - 5 <= y0 <= top_y + 5 and
                                left_x - 5 <= x1 <= right_x + 5 and
                                bottom_y - 5 <= y1 <= top_y + 5):
                                contained_texts.append(elem)
                        
                        # If we have text inside, this could be a table
                        if contained_texts:
                            potential_tables.append({
                                "bbox": (left_x, bottom_y, right_x, top_y),
                                "text_elements": contained_texts
                            })
        
        # Convert potential tables to actual table structures
        for pot_table in potential_tables:
            table_content = self._organize_text_into_table(pot_table["text_elements"], pot_table["bbox"])
            if table_content and len(table_content) >= 2:  # Require at least 2 rows
                tables.append({
                    "content": table_content,
                    "bbox": pot_table["bbox"]
                })
        
        return tables
    
    def _detect_tables_by_alignment(self, page: LTPage, text_elements: List[Dict], lines: List[Tuple]) -> List[Dict]:
        """
        Detect tables by analyzing text alignment patterns.
        This is used for borderless tables.
        """
        tables = []
        
        # Convert text elements to a format for alignment analysis
        blocks = []
        for elem in text_elements:
            x0, y0, x1, y1 = elem["bbox"]
            text = elem["content"]
            # Format: (x0, y0, x1, y1, text, type_indicator, group_id)
            blocks.append((x0, y0, x1, y1, text, None, 0))
        
        # Group texts that appear to be in rows (similar y positions)
        rows = self._group_texts_into_rows(blocks)
        
        # Identify column boundaries by analyzing text positions
        column_x_positions = self._identify_columns(rows)
        
        # If we have a tabular structure with consistent columns
        if len(rows) >= 2 and len(column_x_positions) >= 2:
            # Determine the table's bounding box
            min_x = min(b[0] for row in rows for b in row)
            max_x = max(b[2] for row in rows for b in row)
            min_y = min(b[1] for row in rows for b in row)
            max_y = max(b[3] for row in rows for b in row)
            
            # Create a structured table content from the rows and columns
            table_content = []
            for row_blocks in rows:
                table_row = []
                # Using the column positions, identify which text belongs to which cell
                for i in range(len(column_x_positions) - 1):
                    col_start = column_x_positions[i]
                    col_end = column_x_positions[i+1]
                    
                    # Collect all text blocks that belong to this cell
                    cell_texts = []
                    for block in row_blocks:
                        x0, _, x1, _ = block[:4]
                        text = block[4]
                        
                        # Block belongs to this column if it overlaps with column boundaries
                        if (col_start <= x0 < col_end) or (col_start < x1 <= col_end) or (x0 <= col_start and x1 >= col_end):
                            cell_texts.append(text)
                    
                    # Join all texts in the cell
                    cell_content = " ".join(cell_texts).strip() if cell_texts else ""
                    table_row.append(cell_content)
                
                table_content.append(table_row)
            
            # Only add if we have meaningful content
            if any(any(cell for cell in row) for row in table_content):
                tables.append({
                    "content": table_content,
                    "bbox": (min_x, min_y, max_x, max_y)
                })
        
        return tables
    
    def _group_texts_into_rows(self, blocks: List[Tuple]) -> List[List[Tuple]]:
        """Group text blocks into rows based on y-coordinate proximity."""
        if not blocks:
            return []
            
        # Sort blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: b[1])
        
        rows = []
        current_row = [sorted_blocks[0]]
        current_y = sorted_blocks[0][1]
        
        for block in sorted_blocks[1:]:
            # If this block is close enough to be in the same row
            if abs(block[1] - current_y) < 10:  # Threshold for y-proximity
                current_row.append(block)
            else:
                # Sort current row by x-coordinate
                current_row.sort(key=lambda b: b[0])
                rows.append(current_row)
                
                # Start new row
                current_row = [block]
                current_y = block[1]
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
        
        return rows
    
    def _identify_columns(self, rows: List[List[Tuple]]) -> List[float]:
        """
        Identify potential column boundaries by analyzing text positions.
        Returns a list of x-positions representing column boundaries.
        """
        if not rows:
            return []
            
        # Collect all x-positions (start and end of each text block)
        x_positions = []
        for row in rows:
            for block in row:
                x_positions.append(block[0])  # x0
                x_positions.append(block[2])  # x1
        
        # Sort unique x-positions
        unique_x = sorted(set(x_positions))
        
        # If we have fewer than 2 positions, we can't form columns
        if len(unique_x) < 2:
            return []
            
        # Group x-positions that are close to each other
        grouped_x = []
        current_group = [unique_x[0]]
        
        for x in unique_x[1:]:
            if x - current_group[-1] < 10:  # Threshold for considering positions as the same column
                current_group.append(x)
            else:
                # Use the average of the group as the column position
                grouped_x.append(sum(current_group) / len(current_group))
                current_group = [x]
        
        # Add the last group
        if current_group:
            grouped_x.append(sum(current_group) / len(current_group))
        
        # We need at least 2 column boundaries
        if len(grouped_x) < 2:
            return []
            
        return grouped_x
    
    def _merge_table_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Merge overlapping table candidates and remove duplicates."""
        if not candidates:
            return []
            
        # Sort candidates by top-left position
        sorted_candidates = sorted(candidates, key=lambda c: (c["bbox"][0], c["bbox"][1]))
        
        merged = []
        for candidate in sorted_candidates:
            # Check if this candidate overlaps significantly with any existing merged table
            if not merged or not self._has_significant_overlap(candidate, merged):
                merged.append(candidate)
        
        return merged
    
    def _has_significant_overlap(self, candidate: Dict, merged_list: List[Dict]) -> bool:
        """Check if candidate has significant overlap with any table in the merged list."""
        x0, y0, x1, y1 = candidate["bbox"]
        cand_area = (x1 - x0) * (y1 - y0)
        
        for merged_table in merged_list:
            m_x0, m_y0, m_x1, m_y1 = merged_table["bbox"]
            
            # Calculate intersection
            intersection_x0 = max(x0, m_x0)
            intersection_y0 = max(y0, m_y0)
            intersection_x1 = min(x1, m_x1)
            intersection_y1 = min(y1, m_y1)
            
            # Check if there is an intersection
            if intersection_x0 < intersection_x1 and intersection_y0 < intersection_y1:
                intersection_area = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
                
                # If overlap is more than 70% of either table's area, consider it significant
                overlap_ratio = intersection_area / min(cand_area, (m_x1 - m_x0) * (m_y1 - m_y0))
                if overlap_ratio > 0.7:
                    return True
        
        return False
    
    def _organize_text_into_table(self, text_elements: List[Dict], bbox: Tuple[float, float, float, float]) -> List[List[str]]:
        """
        Organize text elements into a structured table with rows and columns.
        Uses the bounding box to determine the table boundaries.
        """
        if not text_elements:
            return []
            
        # Extract table boundaries
        table_x0, table_y0, table_x1, table_y1 = bbox
        
        # Group text elements by rows based on y-position
        rows = []
        sorted_by_y = sorted(text_elements, key=lambda e: e["bbox"][1])
        
        current_row = [sorted_by_y[0]]
        current_y = sorted_by_y[0]["bbox"][1]
        
        for elem in sorted_by_y[1:]:
            y_mid = (elem["bbox"][1] + elem["bbox"][3]) / 2  # y-center of the text
            
            # If this element is close enough to be in the same row
            if abs(y_mid - current_y) < 15:  # Adjusted threshold for row grouping
                current_row.append(elem)
            else:
                # Sort current row by x-position and add to rows
                current_row.sort(key=lambda e: e["bbox"][0])
                rows.append(current_row)
                
                # Start new row
                current_row = [elem]
                current_y = y_mid
        
        # Add the last row
        if current_row:
            current_row.sort(key=lambda e: e["bbox"][0])
            rows.append(current_row)
        
        # Now determine column boundaries by analyzing all rows
        all_x_positions = []
        for row in rows:
            for elem in row:
                all_x_positions.append(elem["bbox"][0])  # Left edge
                all_x_positions.append(elem["bbox"][2])  # Right edge
        
        # If no positions detected, return empty
        if not all_x_positions:
            return []
            
        # Determine column boundaries using clustering
        column_boundaries = self._cluster_x_positions(all_x_positions)
        
        # Build the table content using the identified rows and columns
        table_content = []
        for row_elements in rows:
            row_content = self._distribute_row_elements_to_columns(row_elements, column_boundaries)
            table_content.append(row_content)
        
        return table_content
    
    def _cluster_x_positions(self, x_positions: List[float]) -> List[float]:
        """
        Cluster x-positions to identify column boundaries.
        Uses a simple distance-based clustering.
        """
        if not x_positions:
            return []
            
        # Sort positions
        sorted_x = sorted(x_positions)
        
        # Identify clusters with a distance threshold
        clusters = []
        current_cluster = [sorted_x[0]]
        
        for x in sorted_x[1:]:
            if x - current_cluster[-1] < 15:  # Threshold for considering positions in the same cluster
                current_cluster.append(x)
            else:
                # Calculate cluster center and add to list
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [x]
        
        # Add the last cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return clusters
    
    def _distribute_row_elements_to_columns(self, row_elements: List[Dict], column_boundaries: List[float]) -> List[str]:
        """
        Distribute text elements in a row to their respective columns.
        Returns a list of strings, one for each column.
        """
        if not row_elements or not column_boundaries:
            return []
            
        # Create empty columns
        columns = ["" for _ in range(len(column_boundaries))]
        
        # Assign each text element to the nearest column
        for elem in row_elements:
            elem_center = (elem["bbox"][0] + elem["bbox"][2]) / 2
            
            # Find the closest column boundary
            closest_idx = min(range(len(column_boundaries)), 
                             key=lambda i: abs(column_boundaries[i] - elem_center))
            
            # Add text to the column (with space if existing text)
            if columns[closest_idx]:
                columns[closest_idx] += " " + elem["content"]
            else:
                columns[closest_idx] = elem["content"]
        
        return columns
    
    def _check_for_nested_tables(self, table: Dict, all_tables: List[Dict]) -> Tuple[List[str], bool]:
        """
        Check if a table contains nested tables by analyzing bounding box containment.
        Returns a list of nested table IDs and a boolean indicating if nesting was found.
        """
        if not all_tables:
            return [], False
            
        parent_bbox = table.get("bbox", (0, 0, 0, 0))
        parent_area = (parent_bbox[2] - parent_bbox[0]) * (parent_bbox[3] - parent_bbox[1])
        
        nested_tables = []
        
        for other_table in all_tables:
            if other_table == table:  # Skip self
                continue
                
            other_bbox = other_table.get("bbox", (0, 0, 0, 0))
            
            # Check if other table is contained within this table
            if (parent_bbox[0] < other_bbox[0] and 
                parent_bbox[1] < other_bbox[1] and 
                parent_bbox[2] > other_bbox[2] and 
                parent_bbox[3] > other_bbox[3]):
                
                # Calculate the area of the other table
                other_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
                
                # Check if the nested table is significantly smaller than the parent
                if other_area < parent_area * self.nested_area_ratio:
                    # Generate a placeholder ID for the nested table
                    nested_id = f"nested_{uuid.uuid4().hex[:8]}"
                    nested_tables.append(nested_id)
        
        return nested_tables, len(nested_tables) > 0
    
    def _process_continuation_tables(self, pdf_data: Dict) -> Dict:
        """
        Identify and merge tables that continue across multiple pages.
        Uses header similarity and positioning to detect continuation.
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
    
    def _check_table_continuation(self, table1: Dict, table2: Dict) -> bool:
        """
        Check if table2 is a continuation of table1 by comparing headers
        and looking for continuation indicators.
        """
        # Get header rows
        if not table1.get("content") or not table2.get("content"):
            return False
            
        header1 = table1["content"][0] if table1["content"] else []
        header2 = table2["content"][0] if table2["content"] else []
        
        # Check for header similarity using string comparison
        header_similarity = self._calculate_header_similarity(header1, header2)
        
        # Check if table1 has continuation indicators
        last_row = table1["content"][-1] if table1["content"] else []
        last_row_text = " ".join(str(cell).lower() for cell in last_row)
        has_continuation_indicator = any(phrase in last_row_text 
                                       for phrase in ["continued", "cont'd", "(continued)", 
                                                     "continues", "to be continued", "(cont)", "(contd)"])
        
        # Also check for positional similarity - tables with same width and column alignment
        positional_similarity = self._check_column_alignment(table1, table2)
        
        # Consider it a continuation if headers are similar or there's a continuation indicator
        # or the column alignment matches very well
        return (header_similarity >= self.header_match_threshold or 
                has_continuation_indicator or
                positional_similarity >= 0.9)
    
    def _calculate_header_similarity(self, header1: List[str], header2: List[str]) -> float:
        """Calculate similarity between two table headers."""
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
            else:
                # Compare using character-by-character similarity
                similarity = self._character_similarity(cell1, cell2)
                if similarity > 0.7:
                    matches += similarity
        
        return matches / total if total > 0 else 0.0
    
    def _character_similarity(self, str1: str, str2: str) -> float:
        """Calculate character-level similarity between two strings."""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
            
        # Basic edit distance calculation
        m, n = len(str1), len(str2)
        
        # Create distance matrix
        dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        # Initialize first row and column
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        
        # Fill the matrix
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Calculate similarity from edit distance
        max_len = max(m, n)
        similarity = 1 - (dp[m][n] / max_len if max_len > 0 else 0)
        
        return similarity
    
    def _check_column_alignment(self, table1: Dict, table2: Dict) -> float:
        """
        Check if two tables have similar column alignment by comparing column widths.
        Returns a similarity score between 0 and 1.
        """
        # Extract the bounding boxes
        bbox1 = table1.get("bbox", (0, 0, 0, 0))
        bbox2 = table2.get("bbox", (0, 0, 0, 0))
        
        # Calculate table widths
        width1 = bbox1[2] - bbox1[0]
        width2 = bbox2[2] - bbox2[0]
        
        # If widths are very different, tables are likely not related
        width_ratio = min(width1, width2) / max(width1, width2) if max(width1, width2) > 0 else 0
        if width_ratio < 0.8:
            return 0.0
            
        # If no content, can't compare further
        if not table1.get("content") or not table2.get("content"):
            return width_ratio
            
        # Compare column counts
        col_count1 = len(table1["content"][0]) if table1["content"] else 0
        col_count2 = len(table2["content"][0]) if table2["content"] else 0
        
        if col_count1 != col_count2:
            return width_ratio * 0.5  # Penalize different column counts
            
        # For tables with same column count, calculate similarity
        return width_ratio