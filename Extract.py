
import io
import logging
import uuid
import re
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import fitz  # PyMuPDF library for PDF extraction

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
    Enhanced PDF extractor with improved table detection and extraction.
    This class handles the extraction of content from PDF files.
    """
    
    def __init__(self, similarity_threshold=0.85, header_match_threshold=0.9, 
                 nested_table_threshold=0.85, nested_area_ratio=0.75,
                 semantic_similarity_threshold=0.80):
        """Initialize the PDF extractor with customizable thresholds."""
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.nested_table_threshold = nested_table_threshold
        self.nested_area_ratio = nested_area_ratio
        self.semantic_similarity_threshold = semantic_similarity_threshold
        self.continuation_tables = []
        
        logger.info("PDFExtractor initialized with the following thresholds:")
        logger.info(f"  similarity_threshold: {similarity_threshold}")
        logger.info(f"  header_match_threshold: {header_match_threshold}")
        logger.info(f"  nested_table_threshold: {nested_table_threshold}")
        logger.info(f"  nested_area_ratio: {nested_area_ratio}")
        logger.info(f"  semantic_similarity_threshold: {semantic_similarity_threshold}")
    
    def extract_pdf_content(self, pdf_content):
        """
        Extract structured content from PDF bytes.
        Returns a structure compatible with the compare.py expectations.
        """
        logger.info("Extracting content from PDF")
        
        # Create a PDF document object
        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        
        # Data structure to store content
        pdf_data = {}
        
        # Process each page
        for page_idx in range(pdf_document.page_count):
            page = pdf_document[page_idx]
            page_num = page_idx + 1  # 1-based page numbering
            
            # Extract text blocks
            text_elements = []
            for text_block in page.get_text("blocks"):
                # Filter for actual text blocks (excluding images)
                if text_block[6] == 0:  # Type 0 means text block
                    text_elements.append({
                        "type": "text",
                        "content": text_block[4],
                        "bbox": text_block[:4]  # x0, y0, x1, y1
                    })
            
            # Extract tables using our enhanced table detection
            tables = self._detect_tables(page)
            
            # Combine all elements
            elements = text_elements + tables
            
            # Store page data
            pdf_data[page_num] = {
                "elements": elements,
                "page_number": page_num,
                "width": page.rect.width,
                "height": page.rect.height,
                "text": page.get_text()
            }
        
        # Post-process to find continuation tables across pages
        pdf_data = self._process_continuation_tables(pdf_data)
        
        logger.info(f"Extracted {len(pdf_data)} pages with content")
        return pdf_data
    
    def _detect_tables(self, page):
        """
        Enhanced table detection algorithm.
        Detects tables even with missing borders.
        """
        tables = []
        
        # Get raw table data from PDF
        raw_tables = self._extract_tables_from_page(page)
        
        # Process each detected table
        for table_idx, raw_table in enumerate(raw_tables):
            # Generate unique ID for the table
            table_id = f"table_{page.number + 1}_{uuid.uuid4().hex[:8]}"
            
            # Get content as list of lists (rows and cells)
            content = raw_table.get("content", [])
            
            # Calculate content hash for matching tables across pages
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Check for nested tables
            nested_tables, has_nested = self._check_for_nested_tables(raw_table, page)
            
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
    
    def _extract_tables_from_page(self, page):
        """
        Extract tables from a page using heuristic analysis.
        This is a placeholder for a more sophisticated algorithm that would:
        1. Look for visual grid patterns
        2. Analyze text alignment to find tabular structures
        3. Identify cells using text positioning
        """
        tables = []
        
        # For demonstration, we'll use a simple approach to find potential tables
        # In a real implementation, we would use more sophisticated algorithms
        
        # Get all text blocks that might form tables (potential rows)
        blocks = page.get_text("blocks")
        
        # Group nearby blocks that may form tables
        potential_tables = self._group_blocks_into_tables(blocks)
        
        # For each potential table group
        for block_group in potential_tables:
            rows = []
            
            # Sort blocks by y-coordinate (top to bottom)
            block_group.sort(key=lambda b: b[1])  # Sort by y0 (top coordinate)
            
            # Group into rows based on y-position
            current_row = []
            current_y = block_group[0][1]
            
            for block in block_group:
                # If this block is on a new row (y position differs significantly)
                if abs(block[1] - current_y) > 10:  # Threshold for new row
                    if current_row:
                        # Sort current row by x-coordinate (left to right)
                        current_row.sort(key=lambda b: b[0])  # Sort by x0 (left coordinate)
                        
                        # Extract text from each block in the row to form cells
                        row_cells = [block[4] for block in current_row]
                        rows.append(row_cells)
                        
                        # Start new row
                        current_row = [block]
                        current_y = block[1]
                else:
                    current_row.append(block)
            
            # Add the last row
            if current_row:
                current_row.sort(key=lambda b: b[0])
                row_cells = [block[4] for block in current_row]
                rows.append(row_cells)
            
            # Calculate bounding box of the entire table
            if block_group:
                x0 = min(block[0] for block in block_group)
                y0 = min(block[1] for block in block_group)
                x1 = max(block[2] for block in block_group)
                y1 = max(block[3] for block in block_group)
                bbox = (x0, y0, x1, y1)
            else:
                bbox = (0, 0, 0, 0)
            
            # Add table if it has at least 2 rows
            if len(rows) >= 2:
                tables.append({
                    "content": rows,
                    "bbox": bbox
                })
        
        return tables
    
    def _group_blocks_into_tables(self, blocks):
        """Group text blocks that may form tables based on spatial proximity."""
        # Filter for text blocks only
        text_blocks = [block for block in blocks if block[6] == 0]
        
        # Simple approach: group blocks that are aligned in a grid pattern
        table_groups = []
        processed = set()
        
        for i, block in enumerate(text_blocks):
            if i in processed:
                continue
                
            # Start a new potential table group
            group = [block]
            processed.add(i)
            
            # Find other blocks that may be part of the same table
            # based on alignment and proximity
            for j, other_block in enumerate(text_blocks):
                if j in processed or j == i:
                    continue
                    
                # Check if horizontally or vertically aligned with the current group
                for group_block in group:
                    h_aligned = (abs(other_block[1] - group_block[1]) < 20 or  # y0 similar
                                abs(other_block[3] - group_block[3]) < 20)    # y1 similar
                    v_aligned = (abs(other_block[0] - group_block[0]) < 20 or  # x0 similar
                                abs(other_block[2] - group_block[2]) < 20)    # x1 similar
                    
                    if h_aligned or v_aligned:
                        group.append(other_block)
                        processed.add(j)
                        break
            
            # Only consider as potential table if we have enough blocks
            if len(group) >= 4:  # Arbitrary threshold
                table_groups.append(group)
        
        return table_groups
    
    def _check_for_nested_tables(self, table, page):
        """Check if a table contains nested tables."""
        # This is a placeholder for a more sophisticated nested table detection algorithm
        # In a real implementation, we would analyze the content and layout
        
        # Placeholder logic - assume no nested tables for this example
        return [], False
    
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
        
        return matches / total if total > 0 else 0.0
