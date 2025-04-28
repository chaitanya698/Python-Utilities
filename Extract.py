"""
Extract.py 
----------
PDF structure-preserving extractor with enhanced table detection.

Key features:
* Advanced header-cache-based continuation detection for multi-page tables
* Enhanced IoU-based nested-table detection
* Improved borderless table detection using text alignment and whitespace analysis
* Content-based table fingerprinting for better matching
* Fixed bounding box validation
"""

import io
import logging
import itertools
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Generator
from collections import defaultdict

import pdfplumber
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ──────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TableData:
    content: List[List[str]]
    bbox: Tuple[float, float, float, float]
    page_number: int
    table_id: str
    parent_id: Optional[str] = None
    is_continuation: bool = False
    continuation_of: Optional[str] = None
    has_nested_tables: bool = False
    nested_tables: List[str] = field(default_factory=list)
    content_hash: str = ""  # New field for content-based matching
    
    def __post_init__(self):
        if self.nested_tables is None:
            self.nested_tables = []
        # Generate content hash for table
        self.content_hash = self._generate_content_hash()
    
    def _generate_content_hash(self) -> str:
        """Generate a hash of table content for matching."""
        if not self.content:
            return ""
        # Join all non-empty cells with a unique separator
        content_str = "⦂".join("∥".join(cell.strip().lower() for cell in row if cell.strip()) 
                             for row in self.content if any(cell.strip() for cell in row))
        # Remove whitespace variations and normalize text
        content_str = re.sub(r'\s+', ' ', content_str).strip()
        return content_str
    
    def merge_continuation(self, other: 'TableData') -> None:
        """Merge a continuation table into this one."""
        # Skip the header row from the continuation table
        if other.content and len(other.content) > 1:
            self.content.extend(other.content[1:])
            # Update content hash after merge
            self.content_hash = self._generate_content_hash()
            logger.info(f"Merged continuation table {other.table_id} into {self.table_id}")
    
    def get_header_signature(self) -> str:
        """Get a signature of the table header for matching."""
        if not self.content or len(self.content) == 0:
            return ""
        return "∥".join(c.strip().lower() for c in self.content[0] if c.strip())


# ──────────────────────────────────────────────────────────────────────
# Extractor
# ──────────────────────────────────────────────────────────────────────
class PDFExtractor:
    def __init__(self, 
                 similarity_threshold: float = 0.85,
                 header_match_threshold: float = 0.9,
                 nested_table_threshold: float = 0.85,
                 nested_area_ratio: float = 0.75,
                 dbscan_eps: float = 5.0,
                 dbscan_min_samples: int = 3):
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.nested_table_threshold = nested_table_threshold
        self.nested_area_ratio = nested_area_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        
        self.table_registry: Dict[str, TableData] = {}
        self._table_counter = 0

        # Header cache for continuation detection
        self.header_cache: Dict[str, TableData] = {}
        
        # Track potential multi-page tables by their signatures
        self.table_signatures: Dict[str, List[Tuple[int, TableData]]] = defaultdict(list)

    # ─── public API ──────────────────────────────────────────────
    def extract_pdf_content(self, raw_bytes: bytes) -> Dict[int, Dict]:
        """
        Extract structured content from PDF.
        
        Return:
            { page_number: {'elements': [ {type:'text'|'table', ...}, … ] } }
        """
        data: Dict[int, Dict] = {}
        tables: List[TableData] = []

        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"Processing PDF with {page_count} pages")
            
            # First pass: extract tables from all pages
            for p_idx, page in enumerate(pdf.pages, start=1):
                logger.info(f"Extracting tables from page {p_idx}/{page_count}")
                page_tables = self._extract_tables(page, p_idx)
                tables.extend(page_tables)
                
                # Register tables by their content signatures for multi-page matching
                for table in page_tables:
                    key_signature = self._get_table_signature(table)
                    self.table_signatures[key_signature].append((p_idx, table))

            # Second pass: process relationships and detect multi-page tables
            logger.info("Processing table relationships and multi-page tables")
            self._process_continuations_and_nesting(tables)
            
            # Third pass: assemble elements in visual order for each page
            for p_idx, page in enumerate(pdf.pages, start=1):
                logger.info(f"Assembling page {p_idx} content")
                # Get tables on this page, excluding continuations already merged
                page_tables = [t for t in tables if t.page_number == p_idx and not t.is_continuation]
                # Extract text elements outside table regions
                text_elements = self._extract_text_outside(page, [t.bbox for t in page_tables])
                
                # Combine and sort all elements by vertical position
                all_elements = page_tables + text_elements
                all_elements.sort(key=lambda el: el.bbox[1] if isinstance(el, TableData) else el["y"])
                
                # Convert to output format
                elements: List[Dict] = []
                for el in all_elements:
                    if isinstance(el, TableData):
                        elements.append({
                            "type": "table",
                            "y": el.bbox[1],
                            "content": el.content,
                            "table_id": el.table_id,
                            "has_nested_tables": el.has_nested_tables,
                            "nested_tables": el.nested_tables,
                            "content_hash": el.content_hash,  # Added for matching
                        })
                    else:
                        elements.append(el)

                data[p_idx] = {"elements": elements}

        return data

    # ─── extraction methods ─────────────────────────────────────
    def _extract_tables(self, page, page_num: int) -> List[TableData]:
        """Extract tables using multiple detection methods."""
        tables: List[TableData] = []
        
        # Method 1: Native pdfplumber table detection (with borders)
        for bbox in page.find_tables():
            # Fix: Validate bounding box is within page dimensions
            valid_bbox = self._validate_bbox(bbox.bbox, page.width, page.height)
            if valid_bbox:
                try:
                    raw = page.crop(valid_bbox).extract_table()
                    if raw:
                        tables.append(self._make_table(raw, valid_bbox, page_num))
                        logger.debug(f"Found bordered table on page {page_num} at {valid_bbox}")
                except Exception as e:
                    logger.warning(f"Error extracting table at {valid_bbox} on page {page_num}: {str(e)}")
        
        # Method 2: Enhanced borderless table detection
        if not tables:
            try:
                borderless_tables = self._detect_borderless_tables(page)
                for bbox, raw in borderless_tables.items():
                    # Fix: Validate bounding box is within page dimensions
                    valid_bbox = self._validate_bbox(bbox, page.width, page.height)
                    if valid_bbox:
                        tables.append(self._make_table(raw, valid_bbox, page_num))
                        logger.debug(f"Detected borderless table on page {page_num} at {valid_bbox}")
            except Exception as e:
                logger.warning(f"Error detecting borderless tables on page {page_num}: {str(e)}")
                
        logger.info(f"Extracted {len(tables)} tables from page {page_num}")
        return tables
        
    def _validate_bbox(self, bbox, page_width, page_height):
        """Validate and fix bounding box to ensure it's within page boundaries."""
        x0, y0, x1, y1 = bbox
        
        # Ensure coordinates are within page boundaries
        x0 = max(0, min(x0, page_width))
        y0 = max(0, min(y0, page_height))
        x1 = max(0, min(x1, page_width))
        y1 = max(0, min(y1, page_height))
        
        # Ensure box has positive dimensions
        if x1 <= x0 or y1 <= y0:
            logger.warning(f"Invalid bounding box dimensions: {bbox}")
            return None
            
        return (x0, y0, x1, y1)
        
    def _detect_borderless_tables(self, page) -> Dict[Tuple[float, float, float, float], List[List[str]]]:
        """
        Enhanced borderless table detection using multiple techniques:
        1. Text alignment clustering
        2. Whitespace analysis
        3. Character position analysis
        """
        # If no chars on page, early return
        chars = page.chars
        if len(chars) < 10:
            return {}
            
        # Step 1: Cluster text lines by y-coordinate for potential rows
        rows = self._cluster_text_by_y(chars)
        if len(rows) < 3:  # Minimum rows for a table
            return {}
            
        # Step 2: Find potential column boundaries through x-coordinate clustering
        x_clusters = self._detect_column_positions(chars)
        if len(x_clusters) < 2:  # Need at least 2 columns
            return {}
            
        # Step 3: Construct table data from the detected structure
        table_data = self._construct_table_from_clusters(chars, rows, x_clusters)
        
        # Step 4: Validate table (check if it has consistent structure)
        if self._is_valid_table_structure(table_data):
            # Calculate bbox from the rows and full page width
            y_min = min(rows.keys())
            y_max = max(rows.keys()) + 10  # Add padding
            
            # Fix: Ensure y_max doesn't exceed page height
            y_max = min(y_max, page.height)
            
            bbox = (0, y_min, page.width, y_max)
            return {bbox: table_data}
        
        return {}
    
    def _cluster_text_by_y(self, chars) -> Dict[float, List]:
        """Group characters by y-coordinate to detect rows."""
        # Extract y-coordinates with appropriate rounding
        y_coords = np.array([[ch['top']] for ch in chars])
        
        # Use DBSCAN for clustering y-coordinates
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(y_coords)
        labels = clustering.labels_
        
        # Group chars by cluster
        rows = {}
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = labels == label
            cluster_chars = [chars[i] for i, belongs in enumerate(mask) if belongs]
            if cluster_chars:
                # Use average y as the key
                avg_y = round(sum(ch['top'] for ch in cluster_chars) / len(cluster_chars), 1)
                rows[avg_y] = cluster_chars
        
        return rows
    
    def _detect_column_positions(self, chars) -> List[float]:
        """Detect potential column boundaries by analyzing x-coordinates."""
        # Extract x-coordinates
        x_coords = np.array([[ch['x0']] for ch in chars])
        
        # Cluster x-coordinates to find column starts
        clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(x_coords)
        labels = clustering.labels_
        
        # Get unique column positions
        columns = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            mask = labels == label
            cluster_chars = [chars[i] for i, belongs in enumerate(mask) if belongs]
            if cluster_chars:
                # Use minimum x as the column start
                min_x = min(ch['x0'] for ch in cluster_chars)
                columns.append(round(min_x, 1))
        
        # Sort columns by x-position
        return sorted(columns)
    
    def _construct_table_from_clusters(self, chars, rows, columns) -> List[List[str]]:
        """Construct table data from detected rows and columns."""
        table = []
        
        # Process rows in vertical order
        for y in sorted(rows.keys()):
            row_chars = rows[y]
            cells = ["" for _ in range(len(columns))]
            
            # Assign each character to nearest column
            for ch in sorted(row_chars, key=lambda c: c['x0']):
                col_idx = self._find_nearest_column(ch['x0'], columns)
                if 0 <= col_idx < len(cells):
                    cells[col_idx] += ch['text']
            
            # Clean up cells
            table.append([c.strip() for c in cells])
        
        return table
    
    def _find_nearest_column(self, x, columns) -> int:
        """Find the nearest column for a given x-coordinate."""
        distances = [abs(x - col) for col in columns]
        return distances.index(min(distances))
    
    def _is_valid_table_structure(self, table_data) -> bool:
        """Validate the detected table structure."""
        if not table_data or len(table_data) < 2:
            return False
            
        # Check if we have consistent number of columns
        col_counts = set(len(row) for row in table_data)
        if len(col_counts) > 1:
            return False
            
        # Check if there's actual content
        content_cells = sum(1 for row in table_data for cell in row if cell.strip())
        if content_cells < 5:  # Arbitrary threshold for minimum content
            return False
            
        return True
    
    def _make_table(self, table: List[List[Any]], bbox, page_num) -> TableData:
        """Create a TableData object from raw table data."""
        # Clean and normalize table content
        cleaned = [[("" if c is None else str(c).strip()) for c in row] for row in table]
        
        # Remove completely empty rows
        cleaned = [row for row in cleaned if any(cell.strip() for cell in row)]
        
        # Create table object
        return TableData(
            content=cleaned,
            bbox=bbox,
            page_number=page_num,
            table_id=self._next_id()
        )

    def _next_id(self):
        """Generate unique table ID."""
        self._table_counter += 1
        return f"table_{self._table_counter}"

    # ─── relationship handling ───────────────────────────────────
    def _process_continuations_and_nesting(self, tables: List[TableData]):
        """Process table relationships including continuations and nesting."""
        # Sort tables by page and then by vertical position
        tables.sort(key=lambda t: (t.page_number, t.bbox[1]))
        
        # First pass: detect continuations based on header similarity
        self._detect_continuations(tables)
        
        # Second pass: detect nested tables
        self._detect_nested_tables(tables)
        
        # Register all tables
        for t in tables:
            self.table_registry[t.table_id] = t
    
    def _detect_continuations(self, tables: List[TableData]):
        """Detect multi-page table continuations."""
        for tbl in tables:
            # Skip empty tables
            if not tbl.content or len(tbl.content) == 0:
                continue
                
            # Get table header signature
            sig = tbl.get_header_signature()
            if not sig:  # Skip tables with empty headers
                continue
            
            # Look for potential parent table
            parent = next((p for hdr, p in self.header_cache.items()
                      if self._header_similarity(sig, hdr) >= self.header_match_threshold
                      and p.page_number < tbl.page_number),  # Must be on earlier page
                     None)
            
            if parent:
                tbl.is_continuation = True
                tbl.continuation_of = parent.table_id
                # Merge this continuation into parent table
                parent.merge_continuation(tbl)
                logger.info(f"Found continuation on page {tbl.page_number} for table from page {parent.page_number}")
            else:
                # No parent found, register this as potential parent
                self.header_cache[sig] = tbl
    
    def _detect_nested_tables(self, tables: List[TableData]):
        """Detect nested tables using enhanced geometric analysis."""
        # Group tables by page for efficiency
        by_page = itertools.groupby(
            sorted(tables, key=lambda t: (t.page_number, -(t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]))),
            key=lambda t: t.page_number
        )
        
        for _, page_tables in by_page:
            page_tables = list(page_tables)
            # Sort by area (largest first) to find parent tables
            page_tables.sort(key=lambda t: -(t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]))
            
            for i, parent in enumerate(page_tables):
                if parent.is_continuation:
                    continue  # Skip continuations
                    
                for child in page_tables[i+1:]:
                    if child.is_continuation:
                        continue  # Skip continuations
                        
                    # Check if child is nested inside parent
                    if self._is_nested(parent.bbox, child.bbox):
                        parent.has_nested_tables = True
                        parent.nested_tables.append(child.table_id)
                        child.parent_id = parent.table_id
                        logger.info(f"Detected nested table {child.table_id} inside {parent.table_id}")

    # ─── utility methods ───────────────────────────────────────────
    def _get_table_signature(self, table: TableData) -> str:
        """Generate a signature for a table based on content."""
        if not table.content or len(table.content) == 0:
            return ""
        
        # Use first row (header) and first column as signature
        header = table.get_header_signature()
        first_col = "∥".join(row[0].strip().lower() for row in table.content[1:] 
                            if row and len(row) > 0 and row[0].strip()) if len(table.content) > 1 else ""
        
        return f"{header}◊{first_col}"
    
    def _header_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two header signatures."""
        if not sig1 or not sig2:
            return 0.0
            
        cells1, cells2 = sig1.split("∥"), sig2.split("∥")
        
        # Count matching cells
        matches = 0
        total = max(len(cells1), len(cells2))
        
        for a, b in itertools.zip_longest(cells1, cells2):
            if a == b:
                matches += 1
            elif a and b:  # Both cells exist but differ
                # Fuzzy match for minor variations
                if self._fuzzy_cell_match(a, b):
                    matches += 0.8  # Partial credit
        
        return matches / total if total > 0 else 0.0
    
    def _fuzzy_cell_match(self, cell1: str, cell2: str) -> bool:
        """Check if two cells are similar despite minor differences."""
        # Normalize text
        c1 = re.sub(r'\s+', ' ', cell1.lower()).strip()
        c2 = re.sub(r'\s+', ' ', cell2.lower()).strip()
        
        # Check if one is substring of the other
        if c1 in c2 or c2 in c1:
            return True
            
        # Allow minor prefix/suffix differences
        min_len = min(len(c1), len(c2))
        if min_len > 3:
            # Check if 80% of characters match
            common_prefix_len = 0
            for i in range(min_len):
                if c1[i] == c2[i]:
                    common_prefix_len += 1
                else:
                    break
                    
            if common_prefix_len / min_len >= 0.8:
                return True
        
        return False
    
    def _is_nested(self, parent: Tuple[float, float, float, float], 
                  child: Tuple[float, float, float, float]) -> bool:
        """Determine if child table is nested inside parent table."""
        # Calculate intersection area
        inter_area = self._inter_area(parent, child)
        child_area = self._area(child)
        parent_area = self._area(parent)
        
        # Child must be mostly contained within parent
        contained = (inter_area / child_area >= self.nested_table_threshold)
        # Child must be significantly smaller than parent
        size_ratio = (child_area / parent_area < self.nested_area_ratio)
        
        return contained and size_ratio
    
    @staticmethod
    def _area(bbox: Tuple[float, float, float, float]) -> float:
        """Calculate area of a bounding box."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def _inter_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """Calculate intersection area between two bounding boxes."""
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, x1 - x0) * max(0, y1 - y0)

    # ─── text extraction ──────────────────────────────────────────
    def _extract_text_outside(self, page, table_bboxes):
        """Extract text elements that don't overlap with tables."""
        elements = []
        for ln in page.extract_text_lines():
            text_bbox = (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
            
            # Check if text overlaps with any table
            if not any(self._inter_area(text_bbox, tb) > 0 for tb in table_bboxes):
                elements.append({
                    "type": "text",
                    "y": ln["top"],
                    "content": ln["text"],
                    "bbox": text_bbox  # Adding bbox for text elements
                })
        
        return elements