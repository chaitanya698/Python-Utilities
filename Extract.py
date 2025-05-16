import io
import os
import logging
import uuid
import hashlib
import functools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# PDF processing libraries
from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTTextBox, LTTextLine, LTPage
from pdfminer.layout import LTRect, LTLine

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_compare.log"),
        logging.StreamHandler()
    ]
)
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
    has_nested_tables: bool = False
    nested_tables: List[str] = None

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


# Memoization decorator for expensive operations
def memoize(func):
    """Memoize decorator for caching expensive function calls"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        if key_hash not in cache:
            cache[key_hash] = func(*args, **kwargs)
            
        return cache[key_hash]
        
    return wrapper


class PDFExtractor:
    """
    Enhanced PDF extractor with improved table detection and extraction.
    Uses parallel processing and optimized algorithms for better performance.
    """
    
    def __init__(self, 
                similarity_threshold: float = 0.85, 
                header_match_threshold: float = 0.9, 
                nested_table_threshold: float = 0.85, 
                nested_area_ratio: float = 0.75,
                max_workers: int = None):
        """Initialize the PDF extractor with customizable thresholds."""
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.nested_table_threshold = nested_table_threshold
        self.nested_area_ratio = nested_area_ratio
        self.continuation_tables = []
        
        # Configure parallel processing
        self.max_workers = max_workers or os.cpu_count()
        
        # Initialize caches
        self._cache = {
            "text_elements": {},
            "lines": {},
            "tables": {},
            "spatial_indices": {},
            "page_text": {},
            "similarity_scores": {}
        }
        
        logger.info("PDFExtractor initialized with the following parameters:")
        logger.info(f"  similarity_threshold: {similarity_threshold}")
        logger.info(f"  header_match_threshold: {header_match_threshold}")
        logger.info(f"  nested_table_threshold: {nested_table_threshold}")
        logger.info(f"  nested_area_ratio: {nested_area_ratio}")
        logger.info(f"  max_workers: {self.max_workers}")
    
    def extract_pdf_content(self, pdf_content: bytes, progress_callback: Callable = None) -> Dict:
        """
        Extract structured content from PDF bytes with parallel processing.
        Returns a structure compatible with the compare.py expectations.
        
        Args:
            pdf_content: The PDF file content as bytes
            progress_callback: Optional callback function for progress reporting
                              Takes a float value from 0.0 to 1.0
        """
        logger.info("Extracting content from PDF using optimized extraction")
        
        # Create an in-memory file object
        pdf_file = io.BytesIO(pdf_content)
        
        # Data structure to store content
        pdf_data = {}
        
        # Extract pages using pdfminer with optimized parameters
        laparams = LAParams(
            line_margin=0.5,
            char_margin=2.0,
            word_margin=0.1,
            boxes_flow=0.5,
            detect_vertical=True
        )
        
        try:
            # Extract all pages at once
            pages = list(extract_pages(pdf_file, laparams=laparams))
            total_pages = len(pages)
            
            if progress_callback:
                progress_callback(0.1)  # 10% progress for initial page extraction
            
            logger.info(f"Extracted {total_pages} pages from PDF")
            
            # Process pages in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Create tasks for parallel processing
                future_to_page = {
                    executor.submit(self._process_page, page_idx, page): page_idx 
                    for page_idx, page in enumerate(pages)
                }
                
                # Collect results
                completed = 0
                for future in as_completed(future_to_page):
                    page_idx = future_to_page[future]
                    page_num = page_idx + 1  # 1-based page numbering
                    
                    try:
                        # Get page data from the future
                        page_data = future.result()
                        pdf_data[page_num] = page_data
                        
                        # Update progress
                        completed += 1
                        if progress_callback:
                            progress_callback(0.1 + 0.7 * (completed / total_pages))
                            
                        logger.debug(f"Processed page {page_num}/{total_pages}")
                    except Exception as e:
                        logger.error(f"Error processing page {page_num}: {str(e)}", exc_info=True)
            
            logger.info(f"Parallel processing completed for {total_pages} pages")
            
            # Post-process to find continuation tables and build relationships
            if progress_callback:
                progress_callback(0.8)  # 80% progress after page processing
                
            pdf_data = self._process_continuation_tables(pdf_data)
            
            # Post-process to identify nested tables
            pdf_data = self._process_nested_tables(pdf_data)
            
            if progress_callback:
                progress_callback(1.0)  # 100% progress when complete
                
            logger.info(f"Extracted {len(pdf_data)} pages with content")
            return pdf_data
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {str(e)}", exc_info=True)
            # Return at least an empty structure on error
            return {}
    
    def _process_page(self, page_idx: int, page: LTPage) -> Dict:
        """
        Process a single page to extract text elements, lines, and tables.
        This method is designed to be run in parallel.
        """
        page_num = page_idx + 1  # 1-based page numbering
        
        try:
            # Step 1: Extract line elements for table border detection
            lines = self._extract_lines(page)
            self._cache["lines"][page_num] = lines
            
            # Step 2: Extract text elements
            text_elements = self._extract_text_elements(page)
            self._cache["text_elements"][page_num] = text_elements
            
            # Step 3: Build spatial index for efficient geometry operations
            spatial_index = self._build_spatial_index(lines, text_elements)
            self._cache["spatial_indices"][page_num] = spatial_index
            
            # Step 4: Detect tables using multiple methods
            tables = self._detect_tables(page, text_elements, lines, spatial_index)
            self._cache["tables"][page_num] = tables
            
            # Step 5: Extract full page text
            page_text = self._extract_page_text(page)
            self._cache["page_text"][page_num] = page_text
            
            # Combine all elements
            elements = text_elements + tables
            
            # Return structured page data
            return {
                "elements": elements,
                "page_number": page_num,
                "width": page.width,
                "height": page.height,
                "text": page_text
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {str(e)}", exc_info=True)
            # Return minimal page data on error
            return {
                "elements": [],
                "page_number": page_num,
                "width": page.width if hasattr(page, 'width') else 0,
                "height": page.height if hasattr(page, 'height') else 0,
                "text": ""
            }
    
    def _extract_text_elements(self, page: LTPage) -> List[Dict]:
        """
        Extract text elements from a page with optimized structure.
        Groups text by blocks for better layout analysis.
        """
        text_elements = []
        
        for element in page:
            if isinstance(element, LTTextBox):
                # Process each text line separately for better granularity
                lines = []
                for line in element:
                    if isinstance(line, LTTextLine):
                        line_text = line.get_text().strip()
                        if line_text:
                            # Store line with its bounding box
                            lines.append({
                                "text": line_text,
                                "bbox": (line.x0, line.y0, line.x1, line.y1)
                            })
                
                # Only process text boxes with content
                if lines:
                    # Get the full text from the element
                    text = element.get_text().strip()
                    if text:
                        # Store text with its bounding box and lines
                        bbox = (element.x0, element.y0, element.x1, element.y1)
                        text_elements.append({
                            "type": "text",
                            "content": text,
                            "bbox": bbox,
                            "lines": lines,
                            # Hash for efficient comparison later
                            "hash": hashlib.md5(text.encode()).hexdigest()
                        })
        
        return text_elements
    
    def _extract_lines(self, page: LTPage) -> List[Dict]:
        """
        Extract line elements that could be table borders with enhanced structure.
        Returns lines with type, coordinates, and additional attributes.
        """
        lines = []
        
        for obj in page:
            # Extract horizontal and vertical lines
            if isinstance(obj, LTRect):
                # Rectangles (often table cells)
                x0, y0, x1, y1 = obj.bbox
                width = x1 - x0
                height = y1 - y0
                
                if width < 2:  # Vertical line
                    lines.append({
                        'type': 'v',  # vertical
                        'x': x0,
                        'y0': y0,
                        'y1': y1,
                        'length': height,
                        'bbox': obj.bbox,
                        'is_rect_edge': True
                    })
                elif height < 2:  # Horizontal line
                    lines.append({
                        'type': 'h',  # horizontal
                        'y': y0,
                        'x0': x0,
                        'x1': x1,
                        'length': width,
                        'bbox': obj.bbox,
                        'is_rect_edge': True
                    })
                else:
                    # Add all 4 sides of the rectangle
                    lines.append({
                        'type': 'h',  # horizontal - bottom
                        'y': y0,
                        'x0': x0,
                        'x1': x1,
                        'length': width,
                        'bbox': (x0, y0, x1, y0+1),
                        'is_rect_edge': True
                    })
                    lines.append({
                        'type': 'h',  # horizontal - top
                        'y': y1,
                        'x0': x0,
                        'x1': x1,
                        'length': width,
                        'bbox': (x0, y1-1, x1, y1),
                        'is_rect_edge': True
                    })
                    lines.append({
                        'type': 'v',  # vertical - left
                        'x': x0,
                        'y0': y0,
                        'y1': y1,
                        'length': height,
                        'bbox': (x0, y0, x0+1, y1),
                        'is_rect_edge': True
                    })
                    lines.append({
                        'type': 'v',  # vertical - right
                        'x': x1,
                        'y0': y0,
                        'y1': y1,
                        'length': height,
                        'bbox': (x1-1, y0, x1, y1),
                        'is_rect_edge': True
                    })
            
            elif isinstance(obj, LTLine):
                # Simple lines
                x0, y0, x1, y1 = obj.bbox
                if abs(x1 - x0) < 2:  # Vertical line
                    lines.append({
                        'type': 'v',  # vertical
                        'x': x0,
                        'y0': min(y0, y1),
                        'y1': max(y0, y1),
                        'length': abs(y1 - y0),
                        'bbox': obj.bbox,
                        'is_rect_edge': False
                    })
                elif abs(y1 - y0) < 2:  # Horizontal line
                    lines.append({
                        'type': 'h',  # horizontal
                        'y': y0,
                        'x0': min(x0, x1),
                        'x1': max(x0, x1),
                        'length': abs(x1 - x0),
                        'bbox': obj.bbox,
                        'is_rect_edge': False
                    })
        
        return lines
    
    def _build_spatial_index(self, lines: List[Dict], text_elements: List[Dict]) -> Dict:
        """
        Build a spatial index for efficient geometry lookups.
        This improves performance when checking intersections and containment.
        """
        spatial_index = {
            'h_lines': sorted(
                [l for l in lines if l['type'] == 'h'], 
                key=lambda x: x['y']
            ),
            'v_lines': sorted(
                [l for l in lines if l['type'] == 'v'], 
                key=lambda x: x['x']
            ),
            'text_by_y': defaultdict(list)
        }
        
        # Group text elements by y-coordinates (rounded to nearest 5 units)
        for element in text_elements:
            y_key = round(element['bbox'][1] / 5) * 5  # Round to nearest 5
            spatial_index['text_by_y'][y_key].append(element)
        
        return spatial_index
    
    def _extract_page_text(self, page: LTPage) -> str:
        """Extract all text from a page as a single string."""
        text_parts = []
        for element in page:
            if isinstance(element, LTTextContainer):
                text_parts.append(element.get_text())
        return "\n".join(text_parts)
    
    def _detect_tables(self, page: LTPage, text_elements: List[Dict], 
                       lines: List[Dict], spatial_index: Dict) -> List[Dict]:
        """
        Enhanced table detection with multiple methods and optimized processing.
        Combines multiple detection approaches for better accuracy.
        """
        # Step 1: Use grid-based detection for implicit tables
        grid_tables = self._detect_tables_by_grid(page, text_elements, spatial_index)
        
        # Step 2: Use explicit border detection
        border_tables = self._detect_tables_by_borders(page, text_elements, lines, spatial_index)
        
        # Step 3: Use text alignment for borderless tables
        alignment_tables = self._detect_tables_by_alignment(page, text_elements, spatial_index)
        
        # Step 4: Merge candidates and remove overlaps
        all_candidates = grid_tables + border_tables + alignment_tables
        merged_tables = self._merge_table_candidates(all_candidates)
        
        # Final processing of tables
        result_tables = []
        for table_idx, table_data in enumerate(merged_tables):
            # Skip tables that are too small
            if (table_data.get('row_count', 0) < 2 or 
                table_data.get('col_count', 0) < 2):
                continue
                
            # Generate unique ID for the table
            table_id = f"table_{page.pageid}_{uuid.uuid4().hex[:8]}"
            
            # Get content as list of lists (rows and cells)
            content = table_data.get("content", [])
            
            # Calculate content hash for matching tables across pages
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in content)
            content_hash = hashlib.md5(content_str.encode()).hexdigest()
            
            # Check for nested tables - will be further processed in post-processing
            table_bbox = table_data.get("bbox", (0, 0, 0, 0))
            
            # Create table element with enhanced properties
            table_element = {
                "type": "table",
                "table_id": table_id,
                "content": content,
                "bbox": table_bbox,
                "content_hash": content_hash,
                "has_nested_tables": False,  # Will be updated in post-processing
                "nested_tables": [],
                "row_count": len(content),
                "col_count": len(content[0]) if content and content[0] else 0
            }
            
            result_tables.append(table_element)
        
        return result_tables
    
    def _detect_tables_by_grid(self, page: LTPage, text_elements: List[Dict], 
                              spatial_index: Dict) -> List[Dict]:
        """
        Detect tables based on text arranged in grid patterns.
        Useful for tables without visible borders.
        """
        tables = []
        
        # Get groups of text with consistent alignment
        text_groups = self._group_texts_by_alignment(text_elements, spatial_index)
        
        for group in text_groups:
            # Only consider groups with at least 2 rows and 2 columns
            if len(group['rows']) < 2 or len(group['columns']) < 2:
                continue
                
            # Get bounding box for this table
            x0 = min(elem['bbox'][0] for elem in group['elements'])
            y0 = min(elem['bbox'][1] for elem in group['elements'])
            x1 = max(elem['bbox'][2] for elem in group['elements'])
            y1 = max(elem['bbox'][3] for elem in group['elements'])
            
            # Convert to structured table content
            content = self._create_table_content_from_grid(
                group['elements'], group['rows'], group['columns']
            )
            
            # Only add if we have actual content
            if content and any(any(cell for cell in row) for row in content):
                tables.append({
                    "content": content,
                    "bbox": (x0, y0, x1, y1),
                    "row_count": len(content),
                    "col_count": len(content[0]) if content else 0
                })
        
        return tables
    
    def _group_texts_by_alignment(self, text_elements: List[Dict], 
                                 spatial_index: Dict) -> List[Dict]:
        """
        Group text elements that appear to be in grid-like patterns.
        Uses spatial clustering to identify rows and columns.
        """
        if not text_elements:
            return []
            
        # Group by y-position first (rows)
        row_clusters = self._cluster_by_position(
            [elem['bbox'][1] for elem in text_elements], 10
        )
        
        # Map y-values to their clusters
        y_to_cluster = {}
        for cluster_idx, cluster in enumerate(row_clusters):
            for y in cluster:
                y_to_cluster[y] = cluster_idx
        
        # Assign elements to row clusters
        rows = defaultdict(list)
        for elem in text_elements:
            y_mid = (elem['bbox'][1] + elem['bbox'][3]) / 2
            # Find closest y-value in our clusters
            closest_y = min(y_to_cluster.keys(), key=lambda y: abs(y - y_mid))
            rows[y_to_cluster[closest_y]].append(elem)
        
        # Now find column alignments by grouping x-positions
        all_x_starts = [elem['bbox'][0] for elem in text_elements]
        col_clusters = self._cluster_by_position(all_x_starts, 15)
        
        # Map x-values to their clusters
        x_to_cluster = {}
        for cluster_idx, cluster in enumerate(col_clusters):
            for x in cluster:
                x_to_cluster[x] = cluster_idx
        
        # Group elements that are aligned in both rows and columns
        grid_groups = []
        
        # For each set of rows that might form a table
        for i in range(len(row_clusters)):
            # Look at consecutive rows (at least 2)
            for j in range(i+1, min(i+10, len(row_clusters))):
                candidate_rows = list(range(i, j+1))
                
                # Collect elements in these rows
                elements = []
                for row_idx in candidate_rows:
                    elements.extend(rows[row_idx])
                
                # Skip if too few elements
                if len(elements) < 4:  # Need at least 2x2 grid
                    continue
                
                # Check column alignment
                elem_x_clusters = set()
                for elem in elements:
                    x_start = elem['bbox'][0]
                    closest_x = min(x_to_cluster.keys(), key=lambda x: abs(x - x_start))
                    elem_x_clusters.add(x_to_cluster[closest_x])
                
                # Need at least 2 columns
                if len(elem_x_clusters) < 2:
                    continue
                
                # This is a potential grid/table
                grid_groups.append({
                    'elements': elements,
                    'rows': candidate_rows,
                    'columns': sorted(list(elem_x_clusters))
                })
        
        return grid_groups
    
    def _cluster_by_position(self, positions: List[float], threshold: float) -> List[List[float]]:
        """
        Cluster positions (x or y coordinates) that are close to each other.
        Uses a simple distance-based clustering algorithm.
        """
        if not positions:
            return []
            
        # Sort the positions
        sorted_pos = sorted(positions)
        
        # Initialize clusters
        clusters = [[sorted_pos[0]]]
        
        # Cluster by threshold
        for pos in sorted_pos[1:]:
            # If this position is close to the last cluster, add to it
            if pos - clusters[-1][-1] <= threshold:
                clusters[-1].append(pos)
            else:
                # Otherwise start a new cluster
                clusters.append([pos])
        
        return clusters
    
    def _create_table_content_from_grid(self, elements: List[Dict], 
                                       row_indices: List[int], 
                                       col_indices: List[int]) -> List[List[str]]:
        """
        Convert a grid of text elements into a structured table content.
        Maps text elements to their respective cells in the table.
        """
        # Create empty table structure
        content = []
        for _ in range(len(row_indices)):
            row = ["" for _ in range(len(col_indices))]
            content.append(row)
        
        # Map elements to their positions in the table
        for elem in elements:
            # Find the row
            y_mid = (elem['bbox'][1] + elem['bbox'][3]) / 2
            closest_row = min(range(len(row_indices)), 
                             key=lambda i: abs(y_mid - sum(e['bbox'][1] for e in elements 
                                                          if e['bbox'][1] in row_indices) / len(row_indices)))
            
            # Find the column
            x_mid = (elem['bbox'][0] + elem['bbox'][2]) / 2
            closest_col = min(range(len(col_indices)), 
                             key=lambda i: abs(x_mid - sum(e['bbox'][0] for e in elements 
                                                          if e['bbox'][0] in col_indices) / len(col_indices)))
            
            # Add the content to the cell (appending if multiple elements in cell)
            current = content[closest_row][closest_col]
            text = elem['content'].strip()
            
            if current:
                content[closest_row][closest_col] = f"{current} {text}"
            else:
                content[closest_row][closest_col] = text
        
        return content
    
    def _detect_tables_by_borders(self, page: LTPage, text_elements: List[Dict], 
                                 lines: List[Dict], spatial_index: Dict) -> List[Dict]:
        """
        Enhanced table detection using explicit border lines.
        Uses spatial indexing for improved performance.
        """
        tables = []
        
        # Skip if insufficient lines to form tables
        if len(lines) < 4:
            return []
            
        # Use horizontal and vertical lines from spatial index
        h_lines = spatial_index['h_lines']
        v_lines = spatial_index['v_lines']
        
        # Skip if insufficient lines
        if len(h_lines) < 2 or len(v_lines) < 2:
            return []
        
        # Find potential table boundaries by looking for rectangular areas
        potential_tables = []
        
        # Group horizontal lines that are close to each other
        h_line_groups = self._cluster_lines_by_position([l['y'] for l in h_lines], 5)
        
        # For each pair of horizontal line groups (potential top and bottom)
        for i, top_group in enumerate(h_line_groups):
            for j, bottom_group in enumerate(h_line_groups):
                # Skip if top is below or same as bottom
                if top_group[0] <= bottom_group[0]:
                    continue
                    
                # Get representative lines
                top_y = sum(top_group) / len(top_group)
                bottom_y = sum(bottom_group) / len(bottom_group)
                
                # Calculate height, skip if too small
                height = top_y - bottom_y
                if height < 20:  # Minimum table height
                    continue
                
                # Find vertical lines that could form left and right boundaries
                # Using a more efficient approach with spatial index
                matching_v_lines = []
                for v_line in v_lines:
                    # Check if vertical line spans between the horizontal lines
                    if v_line['y0'] <= bottom_y + 5 and v_line['y1'] >= top_y - 5:
                        matching_v_lines.append(v_line)
                
                # Need at least 2 vertical lines for a table
                if len(matching_v_lines) < 2:
                    continue
                
                # Group vertical lines that are close to each other
                v_line_groups = self._cluster_lines_by_position([l['x'] for l in matching_v_lines], 5)
                
                # For each pair of vertical line groups (potential left and right)
                for k, left_group in enumerate(v_line_groups):
                    for m, right_group in enumerate(v_line_groups):
                        # Skip if left is to the right or same as right
                        if left_group[0] >= right_group[0]:
                            continue
                            
                        # Get representative lines
                        left_x = sum(left_group) / len(left_group)
                        right_x = sum(right_group) / len(right_group)
                        
                        # Calculate width, skip if too small
                        width = right_x - left_x
                        if width < 20:  # Minimum table width
                            continue
                        
                        # Define table bounding box
                        table_bbox = (left_x, bottom_y, right_x, top_y)
                        
                        # Check if this area contains text elements
                        contained_texts = self._get_elements_in_bbox(text_elements, table_bbox, 5)
                        
                        # If we have text inside, this could be a table
                        if contained_texts:
                            potential_tables.append({
                                "bbox": table_bbox,
                                "text_elements": contained_texts
                            })
        
        # Convert potential tables to actual table structures
        for pot_table in potential_tables:
            # Use improved method to organize text into a structured table
            table_content = self._organize_text_into_table_optimized(
                pot_table["text_elements"], 
                pot_table["bbox"]
            )
            
            # Only add if we have meaningful content with enough rows
            if (table_content and len(table_content) >= 2 and
                any(any(cell for cell in row) for row in table_content)):
                tables.append({
                    "content": table_content,
                    "bbox": pot_table["bbox"],
                    "row_count": len(table_content),
                    "col_count": len(table_content[0]) if table_content else 0
                })
        
        return tables
    
    def _cluster_lines_by_position(self, positions: List[float], threshold: float) -> List[List[float]]:
        """
        Cluster line positions (x or y coordinates) that are close to each other.
        Returns list of clusters where each cluster is a list of positions.
        """
        if not positions:
            return []
            
        # Sort positions
        sorted_pos = sorted(positions)
        
        # Initialize clusters
        clusters = [[sorted_pos[0]]]
        
        # Cluster by threshold
        for pos in sorted_pos[1:]:
            # If close to the last cluster, add to it
            if pos - clusters[-1][-1] <= threshold:
                clusters[-1].append(pos)
            else:
                # Otherwise start a new cluster
                clusters.append([pos])
        
        return clusters
    
    def _get_elements_in_bbox(self, elements: List[Dict], bbox: Tuple[float, float, float, float], 
                             margin: float = 0) -> List[Dict]:
        """
        Get elements that are contained within the given bounding box.
        Uses a margin for more flexible containment testing.
        """
        x0, y0, x1, y1 = bbox
        contained = []
        
        for elem in elements:
            ex0, ey0, ex1, ey1 = elem["bbox"]
            
            # Check if element is inside the bbox with margin
            if (x0 - margin <= ex0 and ex1 <= x1 + margin and
                y0 - margin <= ey0 and ey1 <= y1 + margin):
                contained.append(elem)
        
        return contained
    
    def _organize_text_into_table_optimized(self, text_elements: List[Dict], 
                                           bbox: Tuple[float, float, float, float]) -> List[List[str]]:
        """
        Organize text elements into a structured table with rows and columns.
        Uses improved clustering for more accurate cell assignment.
        """
        if not text_elements:
            return []
            
        # Extract table boundaries
        table_x0, table_y0, table_x1, table_y1 = bbox
        
        # Group text elements by rows using y-coordinates
        elements_by_y = self._cluster_elements_by_y(text_elements)
        rows = [elements for _, elements in sorted(elements_by_y.items())]
        
        # If no rows detected, return empty
        if not rows:
            return []
        
        # Find column boundaries by analyzing x-coordinates
        col_boundaries = self._detect_column_boundaries(rows)
        
        # Create empty table structure
        table_content = []
        
        # Process each row
        for row_elements in rows:
            row_content = ["" for _ in range(len(col_boundaries) - 1)]
            
            # Assign each text element to the appropriate column
            for elem in row_elements:
                x_center = (elem["bbox"][0] + elem["bbox"][2]) / 2
                
                # Find which column this element belongs to
                for col_idx in range(len(col_boundaries) - 1):
                    if col_boundaries[col_idx] <= x_center < col_boundaries[col_idx + 1]:
                        # Add text to the column (with space if not empty)
                        curr = row_content[col_idx]
                        text = elem["content"].strip()
                        
                        if curr:
                            row_content[col_idx] = f"{curr} {text}"
                        else:
                            row_content[col_idx] = text
                        
                        break
            
            table_content.append(row_content)
        
        return table_content
    
    def _cluster_elements_by_y(self, elements: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Cluster text elements by their y-coordinates to identify rows.
        Returns a dictionary mapping cluster indices to lists of elements.
        """
        # Sort elements by y-coordinate (top to bottom)
        sorted_by_y = sorted(elements, key=lambda e: -(e["bbox"][1] + e["bbox"][3]) / 2)
        
        clusters = {}
        cluster_idx = 0
        
        if not sorted_by_y:
            return clusters
            
        # Start first cluster
        current_cluster = [sorted_by_y[0]]
        current_y = (sorted_by_y[0]["bbox"][1] + sorted_by_y[0]["bbox"][3]) / 2
        
        # Process remaining elements
        for elem in sorted_by_y[1:]:
            y_center = (elem["bbox"][1] + elem["bbox"][3]) / 2
            
            # If this element is close enough to current cluster
            if abs(y_center - current_y) < 12:  # Adjusted threshold
                current_cluster.append(elem)
                # Update cluster y as average
                current_y = sum((e["bbox"][1] + e["bbox"][3]) / 2 for e in current_cluster) / len(current_cluster)
            else:
                # Finalize current cluster and start new one
                clusters[cluster_idx] = current_cluster
                cluster_idx += 1
                
                current_cluster = [elem]
                current_y = y_center
        
        # Add the last cluster
        if current_cluster:
            clusters[cluster_idx] = current_cluster
        
        return clusters
    
    def _detect_column_boundaries(self, rows: List[List[Dict]]) -> List[float]:
        """
        Detect column boundaries by analyzing the distribution of text element edges.
        Returns a list of x-coordinates representing column boundaries.
        """
        # Collect all x-positions (start and end of each text element)
        x_positions = []
        for row in rows:
            for elem in row:
                x_positions.append(elem["bbox"][0])  # Left edge
                x_positions.append(elem["bbox"][2])  # Right edge
        
        # If no positions detected, return default
        if not x_positions:
            return [0, 100]  # Default column boundary
        
        # Find peaks in the distribution of x-positions
        # This uses a simplified peak detection algorithm
        hist = defaultdict(int)
        bin_size = 10  # Adjust for sensitivity
        
        for x in x_positions:
            bin_idx = int(x / bin_size) * bin_size
            hist[bin_idx] += 1
        
        # Get histogram peaks
        peaks = []
        bins = sorted(hist.keys())
        
        for i in range(1, len(bins) - 1):
            if (hist[bins[i]] > hist[bins[i-1]] and 
                hist[bins[i]] > hist[bins[i+1]] and
                hist[bins[i]] > 2):  # Minimum peak height
                peaks.append(bins[i])
        
        # Add boundaries at the edges
        min_x = min(x_positions) - bin_size
        max_x = max(x_positions) + bin_size
        
        # Create column boundaries
        boundaries = [min_x] + peaks + [max_x]
        boundaries.sort()
        
        # Ensure minimum column width
        filtered_boundaries = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered_boundaries[-1] >= 20:  # Minimum column width
                filtered_boundaries.append(b)
        
        return filtered_boundaries
    
    def _detect_tables_by_alignment(self, page: LTPage, text_elements: List[Dict],
                                   spatial_index: Dict) -> List[Dict]:
        """
        Detect tables by analyzing text alignment patterns.
        Useful for borderless tables with consistent alignment.
        """
        tables = []
        
        # Convert text elements to a format for alignment analysis
        blocks = []
        for elem in text_elements:
            x0, y0, x1, y1 = elem["bbox"]
            text = elem["content"]
            # Format: (x0, y0, x1, y1, text, type_indicator, element_id)
            blocks.append((x0, y0, x1, y1, text, None, id(elem)))
        
        # Group texts that appear to be in rows (similar y positions)
        rows = self._group_texts_into_rows(blocks)
        
        # Identify column boundaries by analyzing text positions
        column_x_positions = self._identify_columns(rows)
        
        # Process when we have a tabular structure
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
                    "bbox": (min_x, min_y, max_x, max_y),
                    "row_count": len(table_content),
                    "col_count": len(table_content[0]) if table_content else 0
                })
        
        return tables
    
    def _group_texts_into_rows(self, blocks: List[Tuple]) -> List[List[Tuple]]:
        """Group text blocks into rows based on y-coordinate proximity."""
        if not blocks:
            return []
            
        # Sort blocks by y-coordinate (top to bottom)
        sorted_blocks = sorted(blocks, key=lambda b: -b[1])
        
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
        
        # If nothing collected, return empty
        if not x_positions:
            return []
        
        # Sort and remove duplicates
        x_positions = sorted(set(x_positions))
        
        # If we have fewer than 2 positions, we can't form columns
        if len(x_positions) < 2:
            return []
            
        # Group x-positions that are close to each other
        grouped_x = self._cluster_x_positions(x_positions, 10)
        
        # We need at least 2 column boundaries
        if len(grouped_x) < 2:
            return []
            
        return grouped_x
    
    def _cluster_x_positions(self, x_positions: List[float], threshold: float) -> List[float]:
        """
        Cluster x-positions that are close to each other.
        Returns a list of representative x-positions for each cluster.
        """
        if not x_positions:
            return []
            
        # Initialize clusters
        clusters = [[x_positions[0]]]
        
        # Group positions within threshold
        for x in x_positions[1:]:
            if x - clusters[-1][-1] < threshold:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        
        # Calculate cluster representatives
        representatives = [sum(cluster) / len(cluster) for cluster in clusters]
        
        return representatives
    
    def _merge_table_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """
        Merge overlapping table candidates and remove duplicates.
        Uses improved spatial analysis for better merging decisions.
        """
        if not candidates:
            return []
            
        # Sort candidates by area (largest first)
        sorted_candidates = sorted(
            candidates, 
            key=lambda c: ((c["bbox"][2] - c["bbox"][0]) * (c["bbox"][3] - c["bbox"][1])),
            reverse=True
        )
        
        merged = []
        for candidate in sorted_candidates:
            # Skip candidates with invalid bounding boxes
            if (candidate["bbox"][0] >= candidate["bbox"][2] or 
                candidate["bbox"][1] >= candidate["bbox"][3]):
                continue
                
            # Check if this candidate overlaps significantly with any existing merged table
            if not self._has_significant_overlap(candidate, merged):
                merged.append(candidate)
        
        return merged
    
    def _has_significant_overlap(self, candidate: Dict, merged_list: List[Dict]) -> bool:
        """
        Check if candidate has significant overlap with any table in the merged list.
        Uses area-based overlap calculation for better accuracy.
        """
        x0, y0, x1, y1 = candidate["bbox"]
        cand_area = (x1 - x0) * (y1 - y0)
        
        if cand_area <= 0:
            return False
        
        for merged_table in merged_list:
            m_x0, m_y0, m_x1, m_y1 = merged_table["bbox"]
            merged_area = (m_x1 - m_x0) * (m_y1 - m_y0)
            
            if merged_area <= 0:
                continue
            
            # Calculate intersection
            intersection_x0 = max(x0, m_x0)
            intersection_y0 = max(y0, m_y0)
            intersection_x1 = min(x1, m_x1)
            intersection_y1 = min(y1, m_y1)
            
            # Check if there is an intersection
            if intersection_x0 < intersection_x1 and intersection_y0 < intersection_y1:
                intersection_area = (intersection_x1 - intersection_x0) * (intersection_y1 - intersection_y0)
                
                # Calculate overlap ratios relative to both tables
                overlap_ratio_candidate = intersection_area / cand_area
                overlap_ratio_merged = intersection_area / merged_area
                
                # If either overlap is significant, consider it overlapping
                if overlap_ratio_candidate > 0.7 or overlap_ratio_merged > 0.7:
                    return True
        
        return False
    
    def _process_continuation_tables(self, pdf_data: Dict) -> Dict:
        """
        Identify and merge tables that continue across multiple pages.
        Uses improved header matching and content analysis.
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
                
                # Check if tables might be continuations
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
            primary_table["row_count"] = len(merged_content)
            
            # Calculate new content hash
            content_str = "\n".join("∥".join(str(cell) for cell in row) for row in merged_content)
            primary_table["content_hash"] = hashlib.md5(content_str.encode()).hexdigest()
        
        return pdf_data
    
    def _check_table_continuation(self, table1: Dict, table2: Dict) -> bool:
        """
        Check if table2 is a continuation of table1 using improved analysis.
        Considers header similarity, content patterns, and positional features.
        """
        # Get header rows
        if not table1.get("content") or not table2.get("content"):
            return False
            
        header1 = table1["content"][0] if table1["content"] else []
        header2 = table2["content"][0] if table2["content"] else []
        
        # Check for header similarity using enhanced comparison
        header_similarity = self._calculate_header_similarity(header1, header2)
        
        # Check if table1 has continuation indicators
        has_continuation_indicator = False
        if table1["content"]:
            last_row = table1["content"][-1]
            last_row_text = " ".join(str(cell).lower() for cell in last_row)
            continuation_phrases = ["continued", "cont'd", "(continued)", "continues", 
                                  "to be continued", "(cont)", "(contd)"]
            has_continuation_indicator = any(phrase in last_row_text for phrase in continuation_phrases)
        
        # Check positional similarity
        positional_similarity = self._check_table_position_similarity(table1, table2)
        
        # Check size and structure similarity
        structure_similarity = self._check_table_structure_similarity(table1, table2)
        
        # Decision logic for continuation
        is_continuation = (
            (header_similarity >= self.header_match_threshold) or
            (has_continuation_indicator and positional_similarity >= 0.7) or
            (structure_similarity >= 0.9 and positional_similarity >= 0.8)
        )
        
        return is_continuation
    
    def _calculate_header_similarity(self, header1: List[str], header2: List[str]) -> float:
        """
        Calculate similarity between two table headers using improved algorithm.
        Uses a combination of exact, partial, and fuzzy matching for better accuracy.
        """
        if not header1 or not header2:
            return 0.0
            
        # Calculate max possible score based on longer header
        total = max(len(header1), len(header2))
        if total == 0:
            return 0.0
            
        # Calculate matching score
        score = 0.0
        
        # For each cell in header1, find the best match in header2
        for i in range(min(len(header1), len(header2))):
            # Skip empty cells
            if not header1[i].strip() or not header2[i].strip():
                continue
                
            # Normalize and compare cell text
            cell1 = str(header1[i]).strip().lower()
            cell2 = str(header2[i]).strip().lower()
            
            if cell1 == cell2:
                # Exact match
                score += 1.0
            elif cell1 in cell2 or cell2 in cell1:
                # Partial match (one is substring of the other)
                score += 0.7
            else:
                # Calculate character-level similarity
                char_sim = self._character_similarity(cell1, cell2)
                if char_sim > 0.7:
                    score += char_sim
        
        return score / total
    
    @memoize
    def _character_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate character-level similarity between two strings.
        Uses memoization for improved performance.
        """
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
    
    def _check_table_position_similarity(self, table1: Dict, table2: Dict) -> float:
        """
        Check if two tables have similar horizontal position.
        Compares x-coordinates and width for position similarity.
        """
        # Extract bounding boxes
        bbox1 = table1.get("bbox", (0, 0, 0, 0))
        bbox2 = table2.get("bbox", (0, 0, 0, 0))
        
        # Calculate dimensions
        width1 = bbox1[2] - bbox1[0]
        width2 = bbox2[2] - bbox2[0]
        x_center1 = (bbox1[0] + bbox1[2]) / 2
        x_center2 = (bbox2[0] + bbox2[2]) / 2
        
        # Calculate similarity scores
        width_ratio = min(width1, width2) / max(width1, width2) if max(width1, width2) > 0 else 0
        position_diff = abs(x_center1 - x_center2) / max(width1, width2) if max(width1, width2) > 0 else 1.0
        position_similarity = max(0, 1 - position_diff)
        
        # Combine scores with weights
        return 0.6 * width_ratio + 0.4 * position_similarity
    
    def _check_table_structure_similarity(self, table1: Dict, table2: Dict) -> float:
        """
        Check if two tables have similar internal structure.
        Compares number of columns and content patterns.
        """
        # Compare column counts
        content1 = table1.get("content", [])
        content2 = table2.get("content", [])
        
        if not content1 or not content2:
            return 0.0
            
        col_count1 = len(content1[0]) if content1 and content1[0] else 0
        col_count2 = len(content2[0]) if content2 and content2[0] else 0
        
        # Column count similarity
        col_sim = min(col_count1, col_count2) / max(col_count1, col_count2) if max(col_count1, col_count2) > 0 else 0
        
        # Content pattern similarity (simplified)
        pattern_sim = 0.8  # Default assumption
        
        if col_count1 > 0 and col_count1 == col_count2:
            # Compare cell length patterns in first row
            len_pattern1 = [len(str(cell)) for cell in content1[0]]
            len_pattern2 = [len(str(cell)) for cell in content2[0]]
            
            # Calculate correlation of length patterns
            if len(len_pattern1) >= 3:  # Need at least 3 points for meaningful pattern
                pattern_sim = self._calculate_pattern_similarity(len_pattern1, len_pattern2)
        
        # Return combined score
        return 0.7 * col_sim + 0.3 * pattern_sim
    
    def _calculate_pattern_similarity(self, pattern1: List[int], pattern2: List[int]) -> float:
        """
        Calculate similarity between two numeric patterns.
        Uses a simplified correlation measure.
        """
        if len(pattern1) != len(pattern2) or len(pattern1) < 2:
            return 0.0
            
        # Normalize patterns
        def normalize(pattern):
            total = sum(pattern)
            if total == 0:
                return [0] * len(pattern)
            return [val / total for val in pattern]
        
        norm1 = normalize(pattern1)
        norm2 = normalize(pattern2)
        
        # Calculate sum of squared differences
        sum_sq_diff = sum((a - b) ** 2 for a, b in zip(norm1, norm2))
        
        # Convert to similarity score (0 to 1)
        return max(0, 1 - min(1, sum_sq_diff))
    
    def _process_nested_tables(self, pdf_data: Dict) -> Dict:
        """
        Process nested tables to build proper hierarchical relationships.
        Identifies tables contained within other tables and updates hierarchy info.
        """
        # First pass: collect all tables
        all_tables = []
        for page_num, page_data in pdf_data.items():
            for elem_idx, element in enumerate(page_data.get("elements", [])):
                if element.get("type") == "table":
                    all_tables.append({
                        "table": element,
                        "page_num": page_num,
                        "elem_idx": elem_idx,
                        "bbox": element.get("bbox", (0, 0, 0, 0))
                    })
        
        # Second pass: identify nested relationships
        for parent_info in all_tables:
            parent = parent_info["table"]
            parent_bbox = parent_info["bbox"]
            parent_page = parent_info["page_num"]
            
            # Find tables nested inside this one
            nested_tables = []
            for child_info in all_tables:
                child = child_info["table"]
                child_bbox = child_info["bbox"]
                child_page = child_info["page_num"]
                
                # Skip self or tables on different pages
                if (child["table_id"] == parent["table_id"] or
                    child_page != parent_page):
                    continue
                
                # Check if child is contained within parent
                if (parent_bbox[0] < child_bbox[0] and
                    parent_bbox[1] < child_bbox[1] and
                    parent_bbox[2] > child_bbox[2] and
                    parent_bbox[3] > child_bbox[3]):
                    
                    # Calculate area ratio for size comparison
                    parent_area = (parent_bbox[2] - parent_bbox[0]) * (parent_bbox[3] - parent_bbox[1])
                    child_area = (child_bbox[2] - child_bbox[0]) * (child_bbox[3] - child_bbox[1])
                    
                    # Only consider tables that are significantly smaller than parent
                    if child_area < parent_area * self.nested_area_ratio:
                        nested_tables.append(child["table_id"])
                        
                        # Update child to mark it as nested
                        child["is_nested"] = True
                        child["parent_table_id"] = parent["table_id"]
            
            # Update parent if it has nested tables
            if nested_tables:
                parent["has_nested_tables"] = True
                parent["nested_tables"] = nested_tables
        
        # Update elements in the original data structure
        for table_info in all_tables:
            page_num = table_info["page_num"]
            elem_idx = table_info["elem_idx"]
            
            # Skip if no changes
            if "is_nested" not in table_info["table"] and not table_info["table"].get("has_nested_tables"):
                continue
                
            # Update the element in place
            pdf_data[page_num]["elements"][elem_idx] = table_info["table"]
        
        return pdf_data
