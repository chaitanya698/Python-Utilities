"""
Extract.py 
----------
Enhanced PDF structure-preserving extractor with advanced table detection.

Key improvements:
* Advanced header-cache-based continuation detection for multi-page tables
* Enhanced IoU-based nested-table detection
* Improved borderless table detection using text alignment and whitespace analysis
* Content-based table fingerprinting for better matching
* Fixed bounding box validation
* Added text alignment detection for borderless tables
* Enhanced multi-page table continuity detection
"""

import io
import logging
import itertools
import re
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Set, Generator
from collections import defaultdict

import pdfplumber
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

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
    content_hash: str = ""  # Field for content-based matching
    semantic_embedding: Optional[np.ndarray] = None  # Field for semantic matching
    
    def __post_init__(self):
        if self.nested_tables is None:
            self.nested_tables = []
        # Generate content hash for table
        self.content_hash = self._generate_content_hash()
    
    # Added methods to make TableData hashable
    def __hash__(self):
        # Use table_id as a unique identifier for hashing
        return hash(self.table_id)
    
    def __eq__(self, other):
        # Define equality based on table_id
        if not isinstance(other, TableData):
            return False
        return self.table_id == other.table_id
    
    def _generate_content_hash(self) -> str:
        """Generate a hash of table content for matching."""
        if not self.content:
            return ""
        # Join all non-empty cells with a unique separator
        content_str = "⦂".join("∥".join(cell.strip().lower() for cell in row if cell.strip()) 
                             for row in self.content if any(cell.strip() for cell in row))
        # Remove whitespace variations and normalize text
        content_str = re.sub(r'\s+', ' ', content_str).strip()
        # Return a deterministic hash
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def merge_continuation(self, other: 'TableData') -> None:
        """Merge a continuation table into this one."""
        # Skip the header row from the continuation table if it exists
        if other.content and len(other.content) > 1:
            # Check if the first row appears to be a header
            if self._is_header_row(other.content[0]):
                # Skip header row when merging
                self.content.extend(other.content[1:])
            else:
                # Include all rows if no header is detected
                self.content.extend(other.content)
                
            # Update content hash after merge
            self.content_hash = self._generate_content_hash()
            logger.info(f"Merged continuation table {other.table_id} into {self.table_id}")
    
    def _is_header_row(self, row: List[str]) -> bool:
        """Determine if a row appears to be a header row."""
        # Check if this row matches our header
        if len(self.content) > 0:
            header_similarity = self._calculate_row_similarity(self.content[0], row)
            if header_similarity > 0.7:  # If 70% similar, likely a header
                return True
        
        # Additional header detection heuristics
        # Headers often have shorter text and may contain common header words
        header_words = ["total", "sum", "average", "count", "id", "name", "date", "amount", "#"]
        has_header_words = any(any(hw in cell.lower() for hw in header_words) for cell in row if cell)
        avg_length = sum(len(cell) for cell in row if cell) / max(1, sum(1 for cell in row if cell))
        
        # Headers typically have shorter text
        return has_header_words or avg_length < 15
    
    def _calculate_row_similarity(self, row1: List[str], row2: List[str]) -> float:
        """Calculate similarity between two rows."""
        # Ensure rows have same length for comparison
        max_len = max(len(row1), len(row2))
        r1 = row1 + [""] * (max_len - len(row1))
        r2 = row2 + [""] * (max_len - len(row2))
        
        # Count matching cells
        matches = sum(1 for i in range(max_len) if r1[i].strip().lower() == r2[i].strip().lower())
        return matches / max_len if max_len > 0 else 0.0
    
    def get_header_signature(self) -> str:
        """Get a signature of the table header for matching."""
        if not self.content or len(self.content) == 0:
            return ""
        return "∥".join(c.strip().lower() for c in self.content[0] if c.strip())
    
    def get_text_representation(self) -> str:
        """Convert table to text representation for semantic matching."""
        if not self.content:
            return ""
        
        # Join cells with spaces, rows with newlines
        return "\n".join(" ".join(cell.strip() for cell in row if cell.strip()) 
                         for row in self.content if any(c.strip() for c in row))
    
    def has_continuation_indicators(self) -> bool:
        """Check if table contains continuation indicators in the last row."""
        if not self.content or len(self.content) < 1:
            return False
            
        # Check the last row for continuation indicators
        last_row = self.content[-1]
        last_row_text = " ".join(cell.lower() for cell in last_row).strip()
        
        # Common continuation phrases
        indicators = [
            "continued", "continued on next page", "continued on following page",
            "(continued)", "(cont.)", "(continued on next page)", "to be continued",
            "see next page", "continues", "(table continues)"
        ]
        
        return any(indicator in last_row_text for indicator in indicators)


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
                 dbscan_min_samples: int = 3,
                 semantic_similarity_threshold: float = 0.80):
        self.similarity_threshold = similarity_threshold
        self.header_match_threshold = header_match_threshold
        self.nested_table_threshold = nested_table_threshold
        self.nested_area_ratio = nested_area_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.semantic_similarity_threshold = semantic_similarity_threshold
        
        # Initialize embedding model for semantic matching
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.table_registry: Dict[str, TableData] = {}
        self._table_counter = 0

        # Header cache for continuation detection
        self.header_cache: Dict[str, TableData] = {}
        
        # Track potential multi-page tables by their signatures
        self.table_signatures: Dict[str, List[Tuple[int, TableData]]] = defaultdict(list)
        
        # Store continuations for post-processing
        self.continuation_tables: List[Tuple[TableData, TableData]] = []
        
        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}

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
                            "content_hash": el.content_hash,
                        })
                    else:
                        elements.append(el)

                data[p_idx] = {"elements": elements}

        return data

    # ─── extraction methods ─────────────────────────────────────
    def _extract_tables(self, page, page_num: int) -> List[TableData]:
        """
        Extract tables using multiple detection methods:
        1. Standard bordered tables
        2. Borderless tables with text alignment
        3. Tables with partial borders
        4. Character alignment tables
        """
        tables: List[TableData] = []
        
        # Method 1: Native pdfplumber table detection (with borders)
        for bbox in page.find_tables():
            # Validate bounding box is within page dimensions
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
        borderless_tables = self._detect_borderless_tables(page)
        for bbox, raw in borderless_tables.items():
            # Validate bounding box
            valid_bbox = self._validate_bbox(bbox, page.width, page.height)
            if valid_bbox:
                tables.append(self._make_table(raw, valid_bbox, page_num))
                logger.debug(f"Detected borderless table on page {page_num} at {valid_bbox}")
        
        # Method 3: Partial border table detection with improved algorithm
        partial_border_tables = self._detect_partial_border_tables(page)
        for bbox, raw in partial_border_tables.items():
            # Validate bounding box
            valid_bbox = self._validate_bbox(bbox, page.width, page.height)
            if valid_bbox:
                tables.append(self._make_table(raw, valid_bbox, page_num))
                logger.debug(f"Detected partial border table on page {page_num} at {valid_bbox}")
        
        # Method 4: NEW - Character alignment-based table detection
        char_aligned_tables = self._detect_character_aligned_tables(page)
        for bbox, raw in char_aligned_tables.items():
            # Validate bounding box
            valid_bbox = self._validate_bbox(bbox, page.width, page.height)
            if valid_bbox and not self._overlaps_existing_table(valid_bbox, tables):
                tables.append(self._make_table(raw, valid_bbox, page_num))
                logger.debug(f"Detected character-aligned table on page {page_num} at {valid_bbox}")
                
        logger.info(f"Extracted {len(tables)} tables from page {page_num}")
        return tables
        
    def _validate_bbox(self, bbox, page_width, page_height):
        """Validate and fix bounding box to ensure it's within page boundaries."""
        if bbox is None:
            return None
            
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
    
    def _overlaps_existing_table(self, bbox, tables):
        """Check if a bounding box significantly overlaps with any existing table."""
        for table in tables:
            intersection = self._intersection_area(bbox, table.bbox)
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if bbox_area > 0 and intersection / bbox_area > 0.7:
                return True
        return False
        
    def _detect_borderless_tables(self, page) -> Dict[Tuple[float, float, float, float], List[List[str]]]:
        """
        Enhanced borderless table detection using multiple techniques:
        1. Text alignment clustering
        2. Whitespace analysis
        3. Character position analysis
        4. Text line analysis
        """
        # Get all text elements on the page
        text_lines = page.extract_text_lines()
        chars = page.chars
        
        if len(chars) < 10:
            return {}
        
        # Step 1: Find potential tabular regions based on text alignment
        potential_regions = self._find_tabular_regions(text_lines)
        
        # Step 2: Analyze each potential region
        borderless_tables = {}
        
        for region_start_idx, region_end_idx in potential_regions:
            region_lines = text_lines[region_start_idx:region_end_idx]
            
            # Skip if region is too small
            if len(region_lines) < 3:  # Require at least 3 rows for a table
                continue
            
            # Calculate region bbox
            region_bbox = (
                min(line["x0"] for line in region_lines),
                min(line["top"] for line in region_lines),
                max(line["x1"] for line in region_lines),
                max(line["bottom"] for line in region_lines)
            )
            
            # Step 3: Use multiple methods for column detection
            # Method A: Standard column boundaries based on text alignment
            column_positions_a = self._find_column_boundaries(region_lines)
            
            # Method B: Whitespace gap analysis
            column_positions_b = self._detect_whitespace_gaps(chars, region_bbox)
            
            # Method C: Character-level analysis
            column_positions_c = self._analyze_character_positions(chars, region_bbox)
            
            # Merge all methods for more robust column detection
            column_positions = self._merge_column_detections([
                column_positions_a, column_positions_b, column_positions_c
            ])
            
            # Step 4: Skip if too few columns detected
            if len(column_positions) < 3:  # Need at least 2 columns (3 positions: start, mid, end)
                continue
            
            # Step 5: Extract table content based on merged column positions
            table_data = self._extract_table_from_region(region_lines, column_positions)
            
            # Step 6: Validate table structure
            if self._is_valid_table_structure(table_data):
                # Add to borderless tables
                borderless_tables[region_bbox] = table_data
        
        return borderless_tables
    
    def _merge_column_detections(self, position_lists):
        """Merge column positions from multiple detection methods."""
        # Flatten all positions
        all_positions = []
        for positions in position_lists:
            all_positions.extend(positions)
            
        if not all_positions:
            return []
            
        # Cluster nearby positions (within 15 pixels)
        positions = sorted(all_positions)
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] < 15:
                current_cluster.append(pos)
            else:
                # Add median of current cluster
                clusters.append(np.median(current_cluster))
                current_cluster = [pos]
                
        # Add final cluster
        if current_cluster:
            clusters.append(np.median(current_cluster))
            
        return sorted(clusters)
    
    def _detect_whitespace_gaps(self, chars, region_bbox):
        """Detect column boundaries based on whitespace gaps."""
        x0, y0, x1, y1 = region_bbox
        
        # Filter chars in the region
        region_chars = [c for c in chars 
                      if c["x0"] >= x0 and c["x1"] <= x1 
                      and c["top"] >= y0 and c["bottom"] <= y1]
        
        if not region_chars:
            return []
        
        # Create horizontal density profile
        width = int(x1 - x0) + 1
        density = [0] * width
        
        for char in region_chars:
            start = max(0, int(char["x0"] - x0))
            end = min(width - 1, int(char["x1"] - x0))
            for i in range(start, end + 1):
                density[i] += 1
        
        # Find significant gaps (areas with zero density)
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i, d in enumerate(density):
            if d == 0 and not in_gap:
                gap_start = i
                in_gap = True
            elif d > 0 and in_gap:
                gap_end = i
                gap_width = gap_end - gap_start
                if gap_width > 10:  # Significant gaps are wider than 10px
                    gaps.append((gap_start, gap_end, gap_width))
                in_gap = False
        
        # Convert significant gaps to column boundaries
        boundaries = [x0]  # Start with left edge
        for gap_start, gap_end, _ in sorted(gaps, key=lambda g: g[2], reverse=True)[:5]:  # Use top 5 widest gaps
            mid = (gap_start + gap_end) / 2
            boundaries.append(x0 + mid)
        boundaries.append(x1)  # End with right edge
        
        return sorted(boundaries)
    
    def _analyze_character_positions(self, chars, region_bbox):
        """Analyze character positions to find column boundaries."""
        x0, y0, x1, y1 = region_bbox
        
        # Filter chars in the region
        region_chars = [c for c in chars 
                      if c["x0"] >= x0 and c["x1"] <= x1 
                      and c["top"] >= y0 and c["bottom"] <= y1]
        
        if not region_chars:
            return []
        
        # Extract x-positions
        x_starts = [c["x0"] for c in region_chars]
        
        # Cluster x-positions to find column starts
        if len(x_starts) > 10:  # Need sufficient data for clustering
            try:
                # Convert to numpy array for DBSCAN
                X = np.array([[x] for x in x_starts])
                
                # Use DBSCAN for clustering
                clustering = DBSCAN(eps=10, min_samples=3).fit(X)
                
                # Get cluster centers
                boundaries = []
                for label in set(clustering.labels_):
                    if label != -1:  # Skip noise
                        cluster_points = X[clustering.labels_ == label]
                        center = np.median(cluster_points)
                        boundaries.append(center[0])
                
                # Add left and right region boundaries
                boundaries = [x0] + sorted(boundaries) + [x1]
                return boundaries
            except Exception as e:
                logger.debug(f"Character clustering failed: {str(e)}")
        
        # Fallback: simple column detection
        return [x0, x1]
        
    def _find_tabular_regions(self, text_lines) -> List[Tuple[int, int]]:
        """
        Enhanced method to find regions in the page that likely contain tables
        based on text alignment patterns and whitespace.
        """
        if not text_lines:
            return []
            
        regions = []
        current_region_start = None
        line_count = len(text_lines)
        
        # Calculate x-position distributions for each line
        x_positions = []
        for line in text_lines:
            words = line.get("words", [])
            if words:
                positions = [word["x0"] for word in words]
                x_positions.append(positions)
            else:
                x_positions.append([])
        
        # Enhanced alignment detection with sliding window
        window_size = 5  # Look at 5 consecutive lines
        
        for i in range(line_count - window_size + 1):
            window_lines = x_positions[i:i+window_size]
            window_lines = [pos for pos in window_lines if pos]  # Skip empty lines
            
            if len(window_lines) < 3:  # Need at least 3 non-empty lines
                continue
                
            # Calculate average alignment score across the window
            alignment_scores = []
            
            for j in range(len(window_lines) - 1):
                for k in range(j + 1, len(window_lines)):
                    score = self._calculate_alignment_score(window_lines[j], window_lines[k])
                    alignment_scores.append(score)
            
            avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0
            
            # If good alignment detected, mark as potential table
            if avg_alignment > 0.6:
                if current_region_start is None:
                    current_region_start = i
            elif current_region_start is not None:
                # End of aligned region
                regions.append((current_region_start, i + window_size - 1))
                current_region_start = None
        
        # Add final region if still open
        if current_region_start is not None:
            regions.append((current_region_start, line_count))
        
        # Merge overlapping or adjacent regions
        if not regions:
            return []
            
        merged_regions = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged_regions[-1]
            
            # If this region overlaps or is adjacent to previous one
            if start <= prev_end + 1:
                # Merge the regions
                merged_regions[-1] = (prev_start, max(prev_end, end))
            else:
                # Add as new region
                merged_regions.append((start, end))
        
        return merged_regions
        
    def _calculate_alignment_score(self, positions1, positions2):
        """Enhanced alignment score calculation between two sets of x-positions."""
        if not positions1 or not positions2:
            return 0
            
        # Create tolerance windows around positions
        tolerance = 15  # pixels
        
        # Calculate alignment score based on position matches with tolerance
        matches = 0
        for pos1 in positions1:
            # Check if any position in positions2 is within tolerance of pos1
            if any(abs(pos1 - pos2) < tolerance for pos2 in positions2):
                matches += 1
        
        # Calculate second way (ensure symmetry)
        matches2 = 0
        for pos2 in positions2:
            if any(abs(pos2 - pos1) < tolerance for pos1 in positions1):
                matches2 += 1
        
        # Use average of both directions to ensure fair comparison
        match_score1 = matches / len(positions1) if positions1 else 0
        match_score2 = matches2 / len(positions2) if positions2 else 0
        
        return (match_score1 + match_score2) / 2
        
    def _find_column_boundaries(self, region_lines):
        """Find column boundaries based on text positions in the region."""
        # Extract word positions from all lines
        all_positions = []
        for line in region_lines:
            words = line.get("words", [])
            for word in words:
                all_positions.append(word["x0"])
        
        if not all_positions:
            return []
            
        # Cluster positions to find column boundaries
        positions_array = np.array([[pos] for pos in all_positions])
        
        try:
            clustering = DBSCAN(eps=15, min_samples=3).fit(positions_array)
            
            # Extract cluster centers
            centers = []
            for label in set(clustering.labels_):
                if label != -1:  # Skip noise
                    cluster_positions = [all_positions[i] for i in range(len(all_positions)) 
                                        if clustering.labels_[i] == label]
                    if cluster_positions:
                        centers.append(np.median(cluster_positions))
            
            # Add right boundary
            if centers:
                max_x1 = max(line["x1"] for line in region_lines)
                centers.append(max_x1)
                
            # Sort centers
            return sorted(centers)
        except Exception as e:
            logger.debug(f"Column boundary clustering failed: {str(e)}")
            # Fallback: return left and right extremes
            return [min(all_positions), max(line["x1"] for line in region_lines)]
        
    def _extract_table_from_region(self, region_lines, column_positions):
        """Enhanced extraction of table content from text region using column positions."""
        if not column_positions or len(column_positions) < 2:
            return []
            
        num_cols = len(column_positions) - 1
        table_data = []
        
        for line in region_lines:
            row = [""] * num_cols
            
            # Assign words to columns with improved handling
            words = sorted(line.get("words", []), key=lambda w: w["x0"])
            
            for word in words:
                # Find which column this word belongs to
                for i in range(num_cols):
                    col_start = column_positions[i]
                    col_end = column_positions[i+1]
                    
                    # Word positioned clearly in this column
                    if col_start <= word["x0"] < col_end:
                        row[i] += word["text"] + " "
                        break
                    # Word spans columns (common with missing borders)
                    elif word["x0"] < col_start and word["x1"] > col_start:
                        # Assign to column where most of the word lies
                        if word["x1"] - col_start > col_start - word["x0"]:
                            row[i] += word["text"] + " "
                        else:
                            row[max(0, i-1)] += word["text"] + " "
                        break
            
            # Trim spaces
            row = [cell.strip() for cell in row]
            table_data.append(row)
        
        return table_data
    
    def _detect_partial_border_tables(self, page) -> Dict[Tuple[float, float, float, float], List[List[str]]]:
        """
        Enhanced detection of tables with partial borders by:
        1. Finding horizontal and vertical lines
        2. Identifying grid structures even with missing lines
        3. Extracting content from cells with more resilient algorithms
        """
        # Extract all lines from the page
        h_lines = page.horizontal_edges
        v_lines = page.vertical_edges
        
        if not h_lines and not v_lines:
            return {}
        
        # Find potential table regions based on line intersections
        potential_tables = self._find_grid_regions(h_lines, v_lines, page)
        
        # Extract tables from regions
        partial_border_tables = {}
        for bbox in potential_tables:
            try:
                # Create more resilient algorithm for partial border table extraction
                table = self._extract_partial_border_table(page, bbox, h_lines, v_lines)
                
                if table:
                    partial_border_tables[bbox] = table
                    
            except Exception as e:
                logger.debug(f"Failed to extract partial border table: {str(e)}")
                
        return partial_border_tables
    
    def _extract_partial_border_table(self, page, bbox, h_lines, v_lines):
        """Enhanced extraction of tables with partial borders."""
        try:
            # First attempt standard extraction with custom settings
            cropped_page = page.crop(bbox)
            if not cropped_page:
                return None
                
            # More tolerant settings
            table_settings = {
                'snap_tolerance': 15,       # More tolerant of alignment issues
                'snap_x_tolerance': 15,     # More tolerant of horizontal alignment
                'snap_y_tolerance': 15,     # More tolerant of vertical alignment
                'join_tolerance': 5,        # More tolerant of joining
                'edge_min_length': 3,       # Accept shorter lines
                'min_words_vertical': 1,    # Accept even minimal content
                'min_words_horizontal': 1,  # Accept even minimal content
                'intersection_tolerance': 5 # More tolerant of imperfect intersections
            }
            
            # Try standard extraction first
            table = cropped_page.extract_table(table_settings)
            
            # If failed, try with even more lenient settings
            if not table:
                table_settings.update({
                    'join_tolerance': 10,
                    'edge_min_length': 1,
                    'intersection_tolerance': 10
                })
                table = cropped_page.extract_table(table_settings)
            
            # If still failed, try to reconstruct table from lines
            if not table:
                table = self._reconstruct_table_from_lines(cropped_page, h_lines, v_lines, bbox)
            
            # If still no table, use text-based extraction as fallback
            if not table:
                table = self._extract_table_by_text_positions(cropped_page)
            
            return table
            
        except Exception as e:
            logger.debug(f"Table extraction error: {str(e)}")
            return None
    
    def _reconstruct_table_from_lines(self, page, h_lines, v_lines, bbox):
        """Reconstruct a table from horizontal and vertical lines."""
        x0, y0, x1, y1 = bbox
        
        # Filter lines within bbox
        bbox_h_lines = [l for l in h_lines 
                      if l["x0"] >= x0 and l["x1"] <= x1 
                      and l["top"] >= y0 and l["bottom"] <= y1]
        
        bbox_v_lines = [l for l in v_lines 
                      if l["x0"] >= x0 and l["x1"] <= x1 
                      and l["top"] >= y0 and l["bottom"] <= y1]
        
        if not bbox_h_lines or not bbox_v_lines:
            return None
        
        # Get y-positions of horizontal lines
        y_positions = sorted(set([l["top"] for l in bbox_h_lines]))
        
        # Get x-positions of vertical lines
        x_positions = sorted(set([l["x0"] for l in bbox_v_lines]))
        
        if len(y_positions) < 2 or len(x_positions) < 2:
            return None
        
        # Create cell grid
        num_rows = len(y_positions) - 1
        num_cols = len(x_positions) - 1
        
        # Extract text within each cell
        cells = []
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                cell_bbox = (
                    x_positions[j],
                    y_positions[i],
                    x_positions[j+1],
                    y_positions[i+1]
                )
                
                # Extract text in this cell
                cell_text = ""
                try:
                    cropped_cell = page.crop(cell_bbox)
                    if cropped_cell:
                        cell_text = cropped_cell.extract_text() or ""
                except:
                    pass
                    
                row.append(cell_text.strip())
            cells.append(row)
            
        return cells
    
    def _detect_character_aligned_tables(self, page) -> Dict[Tuple[float, float, float, float], List[List[str]]]:
        """
        Detect tables based on character alignment patterns.
        This is especially effective for tables with no borders at all.
        """
        # Get all characters on the page
        chars = page.chars
        
        if len(chars) < 20:  # Need sufficient characters
            return {}
            
        # Group characters by their y-positions (lines)
        y_tolerance = 3  # pixels
        lines = defaultdict(list)
        
        for char in chars:
            y_pos = round(char["top"] / y_tolerance) * y_tolerance
            lines[y_pos].append(char)
            
        # Sort lines by y-position
        sorted_lines = sorted(lines.items())
        
        # Find regions with consistent character alignment patterns
        regions = []
        current_region = None
        
        for i, (y, line_chars) in enumerate(sorted_lines):
            # Skip lines with too few characters
            if len(line_chars) < 5:
                if current_region:
                    regions.append(current_region)
                    current_region = None
                continue
                
            # Sort characters in this line
            line_chars.sort(key=lambda c: c["x0"])
            
            # Get x-positions
            x_positions = [c["x0"] for c in line_chars]
            
            # Check alignment with previous lines if in a region
            if current_region:
                # Get previous line to compare
                prev_y, prev_chars = sorted_lines[current_region[0]]
                prev_chars.sort(key=lambda c: c["x0"])
                prev_x_positions = [c["x0"] for c in prev_chars]
                
                # Calculate alignment score
                alignment_score = self._calculate_alignment_score(x_positions, prev_x_positions)
                
                if alignment_score > 0.6:
                    # Continue current region
                    current_region[1] = i
                else:
                    # End current region and start new one
                    regions.append(current_region)
                    current_region = [i, i]
            else:
                # Start new region
                current_region = [i, i]
        
        # Add final region if exists
        if current_region:
            regions.append(current_region)
            
        # Convert regions to tables
        char_aligned_tables = {}
        
        for start_idx, end_idx in regions:
            if end_idx - start_idx < 2:  # Need at least 3 rows
                continue
                
            # Get region lines
            region_lines = sorted_lines[start_idx:end_idx+1]
            
            # Calculate region bbox
            region_chars = []
            for _, chars in region_lines:
                region_chars.extend(chars)
                
            if not region_chars:
                continue
                
            x0 = min(c["x0"] for c in region_chars)
            y0 = min(c["top"] for c in region_chars)
            x1 = max(c["x1"] for c in region_chars)
            y1 = max(c["bottom"] for c in region_chars)
            
            region_bbox = (x0, y0, x1, y1)
            
            # Analyze character positions to find columns
            x_clusters = self._cluster_character_positions(region_chars)
            
            if len(x_clusters) < 3:  # Need at least 2 columns
                continue
                
            # Extract table content
            table_data = self._extract_table_from_char_positions(region_lines, x_clusters)
            
            if self._is_valid_table_structure(table_data):
                char_aligned_tables[region_bbox] = table_data
        
        return char_aligned_tables
    
    def _cluster_character_positions(self, chars):
        """Cluster character positions to find column boundaries."""
        # Extract starting positions
        x_positions = [c["x0"] for c in chars]
        
        if len(x_positions) < 10:
            return []
            
        try:
            # Use DBSCAN for clustering
            X = np.array([[x] for x in x_positions])
            clustering = DBSCAN(eps=10, min_samples=3).fit(X)
            
            # Get cluster centers
            centers = []
            for label in set(clustering.labels_):
                if label != -1:  # Skip noise
                    cluster_points = X[clustering.labels_ == label]
                    centers.append(float(np.median(cluster_points)))
            
            # Add right boundary
            if centers:
                centers.append(max(c["x1"] for c in chars))
                
            # Sort and return
            return sorted(centers)
            
        except Exception as e:
            logger.debug(f"Character clustering failed: {str(e)}")
            return []
    
    def _extract_table_from_char_positions(self, region_lines, column_positions):
        """Extract table content based on character position clusters."""
        num_cols = len(column_positions) - 1
        table_data = []
        
        for _, line_chars in region_lines:
            row = [""] * num_cols
            
            # Sort characters by x-position
            line_chars.sort(key=lambda c: c["x0"])
            
            for char in line_chars:
                # Find which column this character belongs to
                for i in range(num_cols):
                    if column_positions[i] <= char["x0"] < column_positions[i+1]:
                        row[i] += char["text"]
                        break
            
            # Add row to table
            table_data.append(row)
            
        return table_data
    
    def _find_grid_regions(self, h_lines, v_lines, page):
        """Enhanced method to find regions that form grid structures even with missing lines."""
        # Find all line intersections to build grid
        intersections = []
        
        # More tolerant intersection detection
        for h in h_lines:
            h_y = (h['top'] + h['bottom']) / 2
            h_x0, h_x1 = h['x0'], h['x1']
            
            for v in v_lines:
                v_x = (v['x0'] + v['x1']) / 2
                v_y0, v_y1 = v['top'], v['bottom']
                
                # Check if lines intersect (with tolerance)
                if (h_x0 - 5 <= v_x <= h_x1 + 5) and (v_y0 - 5 <= h_y <= v_y1 + 5):
                    intersections.append((v_x, h_y))
        
        if len(intersections) < 4:  # Need at least 4 intersections for a table
            return []
        
        # Use DBSCAN to cluster intersections
        if len(intersections) >= 10:
            try:
                # Convert to numpy array
                X = np.array(intersections)
                
                # Cluster x-coordinates
                x_clustering = DBSCAN(eps=15, min_samples=2).fit(X[:, 0].reshape(-1, 1))
                x_labels = set(x_clustering.labels_)
                
                # Cluster y-coordinates
                y_clustering = DBSCAN(eps=15, min_samples=2).fit(X[:, 1].reshape(-1, 1))
                y_labels = set(y_clustering.labels_)
                
                # Only proceed if we found good clusters
                if len(x_labels) > 2 and len(y_labels) > 2:
                    # Get cluster centers
                    x_centers = []
                    for label in x_labels:
                        if label != -1:  # Skip noise
                            cluster_points = X[x_clustering.labels_ == label, 0]
                            x_centers.append(float(np.median(cluster_points)))
                    
                    y_centers = []
                    for label in y_labels:
                        if label != -1:  # Skip noise
                            cluster_points = X[y_clustering.labels_ == label, 1]
                            y_centers.append(float(np.median(cluster_points)))
                    
                    # Create grid region
                    if x_centers and y_centers:
                        region = (
                            min(x_centers) - 10,
                            min(y_centers) - 10,
                            max(x_centers) + 10,
                            max(y_centers) + 10
                        )
                        return [region]
            except Exception as e:
                logger.debug(f"Grid clustering failed: {str(e)}")
        
        # Fallback to traditional approach
        # Cluster intersections to find grid cells
        x_coords = [p[0] for p in intersections]
        y_coords = [p[1] for p in intersections]
        
        # Cluster x and y coordinates
        x_clusters = self._cluster_coordinates(x_coords)
        y_clusters = self._cluster_coordinates(y_coords)
        
        if len(x_clusters) < 2 or len(y_clusters) < 2:
            return []
            
        # Create a region that encompasses the entire grid
        region = (
            min(x_clusters) - 5,
            min(y_clusters) - 5,
            max(x_clusters) + 5,
            max(y_clusters) + 5
        )
        
        return [region]
    
    def _cluster_coordinates(self, coords):
        """Cluster coordinates to find distinct positions."""
        if not coords:
            return []
            
        # Sort coordinates
        sorted_coords = sorted(coords)
        
        # Group close coordinates
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for coord in sorted_coords[1:]:
            if coord - current_cluster[-1] < 5:  # Threshold for same position
                current_cluster.append(coord)
            else:
                # Add average of current cluster and start new one
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [coord]
        
        # Add final cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
            
        return clusters
    
    def _extract_table_by_text_positions(self, page):
        """Improved extraction of table content based on text positions."""
        # Get all words on the page
        words = page.extract_words()
        
        if not words:
            return None
            
        # Improved position clustering for more accurate extraction
        # Cluster y-positions for rows
        y_positions = [word['top'] for word in words]
        
        if not y_positions:
            return None
            
        try:
            # Use DBSCAN to cluster y-positions
            Y = np.array([[y] for y in y_positions])
            y_clustering = DBSCAN(eps=5, min_samples=2).fit(Y)
            
            # Extract row positions
            y_centers = []
            for label in set(y_clustering.labels_):
                if label != -1:  # Skip noise
                    cluster_points = Y[y_clustering.labels_ == label]
                    y_centers.append(float(np.median(cluster_points)))
            
            # Sort by y-position
            y_centers.sort()
        except Exception:
            # Fallback to simple clustering
            y_centers = self._cluster_coordinates(y_positions)
        
        # Cluster x-positions for columns
        x_positions = [word['x0'] for word in words]
        
        try:
            # Use DBSCAN to cluster x-positions
            X = np.array([[x] for x in x_positions])
            x_clustering = DBSCAN(eps=15, min_samples=2).fit(X)
            
            # Extract column positions
            x_centers = []
            for label in set(x_clustering.labels_):
                if label != -1:  # Skip noise
                    cluster_points = X[x_clustering.labels_ == label]
                    x_centers.append(float(np.median(cluster_points)))
            
            # Sort by x-position
            x_centers.sort()
        except Exception:
            # Fallback to simple clustering
            x_centers = self._cluster_coordinates(x_positions)
        
        if len(y_centers) < 2 or len(x_centers) < 2:
            return None
            
        # Create empty table
        table = [["" for _ in range(len(x_centers))] for _ in range(len(y_centers))]
        
        # More accurate cell assignment
        for word in words:
            # Find row index with improved accuracy
            r_idx = None
            min_dist = float('inf')
            for i, y in enumerate(y_centers):
                dist = abs(word['top'] - y)
                if dist < min_dist and dist < 10:  # 10px tolerance
                    min_dist = dist
                    r_idx = i
                    
            # Find column index with improved accuracy
            c_idx = None
            min_dist = float('inf')
            for i, x in enumerate(x_centers):
                dist = abs(word['x0'] - x)
                if dist < min_dist and dist < 20:  # 20px tolerance
                    min_dist = dist
                    c_idx = i
                    
            if r_idx is not None and c_idx is not None and 0 <= r_idx < len(table) and 0 <= c_idx < len(table[0]):
                table[r_idx][c_idx] += word['text'] + " "
        
        # Clean cells
        for r in range(len(table)):
            for c in range(len(table[0])):
                table[r][c] = table[r][c].strip()
                
        return table
    
    def _is_valid_table_structure(self, table_data) -> bool:
        """Enhanced validation of the detected table structure."""
        if not table_data or len(table_data) < 2:
            return False
            
        # Check if we have consistent number of columns
        col_counts = [len(row) for row in table_data]
        
        # Calculate mode column count
        if not col_counts:
            return False
            
        mode_count = max(set(col_counts), key=col_counts.count)
        
        # Check if most rows have the expected column count
        consistent_rows = sum(1 for count in col_counts if abs(count - mode_count) <= 1)
        consistency_ratio = consistent_rows / len(col_counts)
        
        if consistency_ratio < 0.7:  # Less than 70% rows have consistent columns
            return False
            
        # Check if there's actual content
        content_cells = sum(1 for row in table_data for cell in row if cell.strip())
        if content_cells < max(4, len(table_data)):  # At least a few cells have content
            return False
            
        # Check content density
        total_cells = sum(len(row) for row in table_data)
        if total_cells > 0:
            content_density = content_cells / total_cells
            if content_density < 0.2:  # Less than 20% of cells have content
                return False
            
        # Check for too many empty rows
        empty_rows = sum(1 for row in table_data if not any(cell.strip() for cell in row))
        if empty_rows > len(table_data) // 2:
            return False
            
        return True
    
    def _make_table(self, table: List[List[Any]], bbox, page_num) -> TableData:
        """Create a TableData object from raw table data."""
        # Clean and normalize table content
        cleaned = [[("" if c is None else str(c).strip()) for c in row] for row in table]
        
        # Remove completely empty rows
        cleaned = [row for row in cleaned if any(cell.strip() for cell in row)]
        
        if not cleaned:
            cleaned = [[""]]
        
        # Ensure all rows have the same number of columns
        max_cols = max(len(row) for row in cleaned) if cleaned else 0
        cleaned = [row + [""] * (max_cols - len(row)) for row in cleaned]
        
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
        
        # Generate semantic embeddings for all tables
        self._generate_table_embeddings(tables)
        
        # First pass: detect continuations using multiple methods
        self._detect_continuations(tables)
        
        # Second pass: detect nested tables
        self._detect_nested_tables(tables)
        
        # Third pass: process the identified continuations
        self._merge_continuation_tables()
        
        # Register all tables
        for t in tables:
            self.table_registry[t.table_id] = t
    
    def _generate_table_embeddings(self, tables: List[TableData]):
        """Generate semantic embeddings for all tables."""
        # Only process tables with content
        tables_with_content = [t for t in tables if t.content and any(any(cell.strip() for cell in row) for row in t.content)]
        
        # Get text representations
        text_representations = [t.get_text_representation() for t in tables_with_content]
        
        # Generate embeddings in batches to avoid memory issues
        batch_size = 32
        num_tables = len(tables_with_content)
        
        for i in range(0, num_tables, batch_size):
            batch_end = min(i + batch_size, num_tables)
            batch_tables = tables_with_content[i:batch_end]
            batch_texts = text_representations[i:batch_end]
            
            try:
                # Generate embeddings
                embeddings = self.embedding_model.encode(batch_texts)
                
                # Assign embeddings to tables
                for j, table in enumerate(batch_tables):
                    table.semantic_embedding = embeddings[j]
                    # Cache embedding
                    self.embedding_cache[table.table_id] = embeddings[j]
            except Exception as e:
                logger.warning(f"Error generating embeddings: {str(e)}")
    
    def _detect_continuations(self, tables: List[TableData]):
        """
        Enhanced multi-page table detection using multiple methods:
        1. Header-based detection
        2. Footer-based detection ("continued on next page")
        3. Pagination detection
        4. Position-based detection
        5. Semantic similarity detection
        6. Content continuity detection
        """
        # Group tables by similar characteristics
        self._group_and_match_tables(tables)
        
        # Header-based continuation detection
        self._detect_header_based_continuations(tables)
        
        # Apply additional continuation detection methods
        self._detect_footer_continuations(tables)
        self._detect_position_based_continuations(tables)
        self._detect_semantic_continuations(tables)
        self._detect_content_continuity(tables)
    
    def _group_and_match_tables(self, tables: List[TableData]):
        """Group tables with similar characteristics for potential matching."""
        # Group by similar column count and width
        width_groups = defaultdict(list)
        col_count_groups = defaultdict(list)
        
        for table in tables:
            if not table.content:
                continue
                
            # Get table width and number of columns
            width = table.bbox[2] - table.bbox[0]
            width_key = int(width / 10) * 10  # Group by 10-unit width increments
            col_count = len(table.content[0]) if table.content else 0
            
            # Add to groups
            width_groups[width_key].append(table)
            col_count_groups[col_count].append(table)
            
        # Store groups for later use
        self.width_groups = width_groups
        self.col_count_groups = col_count_groups
    
    def _detect_header_based_continuations(self, tables: List[TableData]):
        """Detect table continuations based on header similarity."""
        processed_tables = set()
        
        for table in tables:
            if table in processed_tables or not table.content or len(table.content) == 0:
                continue
                
            # Get table header signature
            header_sig = table.get_header_signature()
            if not header_sig:  # Skip tables with empty headers
                continue
            
            # Find tables on subsequent pages with similar headers
            candidates = []
            for other in tables:
                if (other != table and other.page_number > table.page_number 
                        and other not in processed_tables
                        and other.content and len(other.content) > 0):
                    
                    other_header = other.get_header_signature()
                    if not other_header:
                        continue
                        
                    # Calculate header similarity
                    similarity = self._header_similarity(header_sig, other_header)
                    
                    # Check for additional similarity factors
                    width_similarity = self._compare_table_widths(table, other)
                    col_count_similarity = (len(table.content[0]) == len(other.content[0])) if table.content and other.content else False
                    
                    # Combine similarity measures
                    combined_similarity = (similarity * 0.6 + width_similarity * 0.2 + (1.0 if col_count_similarity else 0.0) * 0.2)
                    
                    if combined_similarity >= self.header_match_threshold:
                        candidates.append((other, combined_similarity))
            
            # Sort candidates by page number (to maintain order) and then similarity
            candidates.sort(key=lambda x: (x[0].page_number, -x[1]))
            
            # Process candidates
            for other, similarity in candidates:
                other.is_continuation = True
                other.continuation_of = table.table_id
                self.continuation_tables.append((table, other))
                processed_tables.add(other)
                logger.info(f"Found header-based continuation on page {other.page_number} for table from page {table.page_number}")
                
                # Register for future continuation chains
                if header_sig:
                    self.header_cache[header_sig] = table
    
    def _compare_table_widths(self, table1, table2):
        """Compare the widths of two tables and return a similarity score (0-1)."""
        width1 = table1.bbox[2] - table1.bbox[0]
        width2 = table2.bbox[2] - table2.bbox[0]
        
        if width1 == 0 or width2 == 0:
            return 0.0
            
        # Calculate ratio of smaller to larger width
        ratio = min(width1, width2) / max(width1, width2)
        return ratio
    
    def _header_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate enhanced similarity between two header signatures."""
        if not sig1 or not sig2:
            return 0.0
            
        cells1, cells2 = sig1.split("∥"), sig2.split("∥")
        
        # Ensure equal length for comparison
        max_len = max(len(cells1), len(cells2))
        c1 = cells1 + [""] * (max_len - len(cells1))
        c2 = cells2 + [""] * (max_len - len(cells2))
        
        # Calculate cell-by-cell similarity with improved matching
        similarity_sum = 0.0
        for a, b in zip(c1, c2):
            a_norm = a.strip().lower()
            b_norm = b.strip().lower()
            
            if not a_norm and not b_norm:
                similarity_sum += 1.0  # Both empty cells match
            elif not a_norm or not b_norm:
                similarity_sum += 0.0  # One empty, one filled = no match
            elif a_norm == b_norm:
                similarity_sum += 1.0  # Exact match
            elif a_norm in b_norm or b_norm in a_norm:
                similarity_sum += 0.8  # Substring match
            else:
                # Use character-level similarity for near matches
                from difflib import SequenceMatcher
                char_sim = SequenceMatcher(None, a_norm, b_norm).ratio()
                if char_sim >= 0.7:
                    similarity_sum += char_sim
        
        return similarity_sum / max_len if max_len > 0 else 0.0
    
    def _detect_footer_continuations(self, tables: List[TableData]):
        """Detect continuations based on footer indicators like 'continued on next page'."""
        # Tables already identified as continuations
        continuation_ids = set(t.table_id for t in tables if t.is_continuation)
        
        # Process each table
        for i, table in enumerate(tables):
            if table.table_id in continuation_ids or not table.content:
                continue
                
            # Check if this table has a continuation indicator
            if table.has_continuation_indicators():
                # Look for a table on the next page
                next_page = table.page_number + 1
                next_page_tables = [t for t in tables 
                                   if t.page_number == next_page 
                                   and t.table_id not in continuation_ids]
                
                if not next_page_tables:
                    continue
                    
                # Find the best matching table on the next page
                best_match = None
                best_score = 0.0
                
                for candidate in next_page_tables:
                    # Calculate similarity based on multiple factors
                    # 1. Similar width
                    width_sim = self._compare_table_widths(table, candidate)
                    
                    # 2. Similar x-position
                    x_pos_diff = abs(table.bbox[0] - candidate.bbox[0])
                    x_pos_sim = max(0, 1.0 - x_pos_diff / 100.0)  # Normalize to 0-1
                    
                    # 3. Similar column count
                    col_sim = 1.0 if (len(table.content[0]) == len(candidate.content[0])) else 0.0
                    
                    # 4. Semantic similarity if available
                    semantic_sim = 0.5  # Default
                    if table.semantic_embedding is not None and candidate.semantic_embedding is not None:
                        semantic_sim = np.dot(table.semantic_embedding, candidate.semantic_embedding)
                    
                    # 5. Position at top of next page
                    top_pos = 1.0 if candidate.bbox[1] < 100 else 0.5  # Prefer tables at top of page
                    
                    # Combine factors with weights
                    score = (width_sim * 0.3 + x_pos_sim * 0.2 + col_sim * 0.2 + 
                            semantic_sim * 0.2 + top_pos * 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_match = candidate
                
                # If good match found, mark as continuation
                if best_match and best_score > 0.6:
                    best_match.is_continuation = True
                    best_match.continuation_of = table.table_id
                    self.continuation_tables.append((table, best_match))
                    continuation_ids.add(best_match.table_id)
                    logger.info(f"Found footer-based continuation on page {best_match.page_number} for table from page {table.page_number}")
    
    def _detect_position_based_continuations(self, tables: List[TableData]):
        """Detect continuations based on position and size similarity."""
        # Tables already identified as continuations
        continuation_ids = set(t.table_id for t in tables if t.is_continuation)
        
        # Group tables by page number
        tables_by_page = defaultdict(list)
        for table in tables:
            if table.table_id not in continuation_ids:
                tables_by_page[table.page_number].append(table)
        
        # Process each page
        for page_num in sorted(tables_by_page.keys())[:-1]:  # Skip last page
            next_page = page_num + 1
            if next_page not in tables_by_page:
                continue
                
            # Tables on current page
            current_page_tables = tables_by_page[page_num]
            
            # Tables on next page
            next_page_tables = tables_by_page[next_page]
            
            # Find tables at bottom of current page
            bottom_tables = self._find_bottom_tables(current_page_tables)
            
            # Find tables at top of next page
            top_tables = self._find_top_tables(next_page_tables)
            
            # Match tables based on width and column count
            for bottom_table in bottom_tables:
                for top_table in top_tables:
                    # Skip if already identified as continuation
                    if top_table.table_id in continuation_ids:
                        continue
                        
                    # Compare width and column count
                    width_sim = self._compare_table_widths(bottom_table, top_table)
                    col_match = (len(bottom_table.content[0]) == len(top_table.content[0])) if bottom_table.content and top_table.content else False
                    
                    # Enhanced horizontal position similarity measure
                    x_pos_diff = abs(bottom_table.bbox[0] - top_table.bbox[0])
                    x_pos_sim = max(0, 1.0 - x_pos_diff / 50.0)  # More strict
                    
                    # Check for semantic similarity
                    semantic_sim = 0.5  # Default
                    if bottom_table.semantic_embedding is not None and top_table.semantic_embedding is not None:
                        semantic_sim = np.dot(bottom_table.semantic_embedding, top_table.semantic_embedding)
                    
                    # Calculate overall match score
                    match_score = width_sim * 0.4 + (1.0 if col_match else 0.0) * 0.3 + x_pos_sim * 0.2 + semantic_sim * 0.1
                    
                    if match_score >= 0.7:
                        top_table.is_continuation = True
                        top_table.continuation_of = bottom_table.table_id
                        self.continuation_tables.append((bottom_table, top_table))
                        continuation_ids.add(top_table.table_id)
                        logger.info(f"Found position-based continuation on page {top_table.page_number} for table from page {bottom_table.page_number}")
    
    def _find_bottom_tables(self, tables, threshold=0.7):
        """Find tables at the bottom of the page."""
        if not tables:
            return []
            
        # Estimate page height using max y coordinate
        max_y = max(table.bbox[3] for table in tables)
        
        # Calculate threshold position (e.g., bottom 30% of page)
        bottom_threshold = max_y * (1 - threshold)
        
        # Return tables with bottom edge in bottom portion of page
        return [table for table in tables if table.bbox[3] > bottom_threshold]
    
    def _find_top_tables(self, tables, threshold=0.3):
        """Find tables at the top of the page."""
        if not tables:
            return []
            
        # Estimate page height using max y coordinate
        max_y = max(table.bbox[3] for table in tables)
        
        # Calculate threshold position (e.g., top 30% of page)
        top_threshold = max_y * threshold
        
        # Return tables with top edge in top portion of page
        return [table for table in tables if table.bbox[1] < top_threshold]
    
    def _detect_semantic_continuations(self, tables: List[TableData]):
        """Detect continuations based on semantic similarity."""
        # Tables already identified as continuations
        continuation_ids = set(t.table_id for t in tables if t.is_continuation)
        
        # Group tables by page
        tables_by_page = defaultdict(list)
        for table in tables:
            if table.table_id not in continuation_ids:
                tables_by_page[table.page_number].append(table)
        
        # Process pages in order
        for page_num in sorted(tables_by_page.keys())[:-1]:  # Skip last page
            next_page = page_num + 1
            if next_page not in tables_by_page:
                continue
            
            for curr_table in tables_by_page[page_num]:
                # Skip tables without semantic embedding
                if curr_table.semantic_embedding is None:
                    continue
                    
                # Check tables on next page
                for next_table in tables_by_page[next_page]:
                    # Skip if already identified as continuation
                    if next_table.table_id in continuation_ids:
                        continue
                        
                    # Skip tables without semantic embedding
                    if next_table.semantic_embedding is None:
                        continue
                    
                    # Calculate semantic similarity
                    similarity = np.dot(curr_table.semantic_embedding, next_table.semantic_embedding)
                    
                    # Check for high semantic similarity and structural match
                    struct_match = len(curr_table.content[0]) == len(next_table.content[0]) if curr_table.content and next_table.content else False
                    
                    if similarity >= self.semantic_similarity_threshold and struct_match:
                        next_table.is_continuation = True
                        next_table.continuation_of = curr_table.table_id
                        self.continuation_tables.append((curr_table, next_table))
                        continuation_ids.add(next_table.table_id)
                        logger.info(f"Found semantic-based continuation on page {next_page} for table from page {page_num}")
    
    def _detect_content_continuity(self, tables: List[TableData]):
        """
        Detect continuations based on content continuity.
        For example, if one table ends with row numbers 1-10 and the next begins with 11-20.
        """
        # Tables already identified as continuations
        continuation_ids = set(t.table_id for t in tables if t.is_continuation)
        
        # Group tables by page
        tables_by_page = defaultdict(list)
        for table in tables:
            if table.table_id not in continuation_ids:
                tables_by_page[table.page_number].append(table)
        
        # Process pages in order
        for page_num in sorted(tables_by_page.keys())[:-1]:  # Skip last page
            next_page = page_num + 1
            if next_page not in tables_by_page:
                continue
                
            for curr_table in tables_by_page[page_num]:
                # Skip tables without content
                if not curr_table.content or len(curr_table.content) < 2:
                    continue
                    
                # Get last row of current table
                last_row = curr_table.content[-1]
                
                for next_table in tables_by_page[next_page]:
                    # Skip if already identified as continuation
                    if next_table.table_id in continuation_ids:
                        continue
                        
                    # Skip tables without content
                    if not next_table.content or len(next_table.content) < 2:
                        continue
                    
                    # Skip header row of next table for comparison
                    next_first_row = next_table.content[1] if len(next_table.content) > 1 else next_table.content[0]
                    
                    # Check for numerical continuity
                    if self._check_numerical_continuity(last_row, next_first_row):
                        next_table.is_continuation = True
                        next_table.continuation_of = curr_table.table_id
                        self.continuation_tables.append((curr_table, next_table))
                        continuation_ids.add(next_table.table_id)
                        logger.info(f"Found content continuity between table on page {page_num} and table on page {next_page}")
                        
                    # Also check for alphabetical continuity (e.g. A-M in first table, N-Z in second)
                    elif self._check_alphabetical_continuity(last_row, next_first_row):
                        next_table.is_continuation = True
                        next_table.continuation_of = curr_table.table_id
                        self.continuation_tables.append((curr_table, next_table))
                        continuation_ids.add(next_table.table_id)
                        logger.info(f"Found alphabetical continuity between table on page {page_num} and table on page {next_page}")
    
    def _check_numerical_continuity(self, row1, row2):
        """
        Enhanced check if two rows show numerical continuity.
        For example, if row1 ends with number X and row2 starts with number X+1.
        """
        # Extract numbers from both rows
        numbers1 = self._extract_numbers(row1)
        numbers2 = self._extract_numbers(row2)
        
        if not numbers1 or not numbers2:
            return False
        
        # Check if last number of row1 is consecutive with first number of row2
        last_num = numbers1[-1]
        first_num = numbers2[0]
        
        # Check for continuity with more tolerance
        return abs(first_num - last_num) <= 3  # Allow small gaps
    
    def _check_alphabetical_continuity(self, row1, row2):
        """
        Check if two rows show alphabetical continuity.
        For example, if row1 has 'K' as last letter and row2 has 'L' as first letter.
        """
        # Extract alphabetical sequences from rows
        alpha1 = self._extract_alphabetical(row1)
        alpha2 = self._extract_alphabetical(row2)
        
        if not alpha1 or not alpha2:
            return False
            
        # Get last letter of first row and first letter of second row
        last_letter = alpha1[-1][-1].upper()
        first_letter = alpha2[0][0].upper()
        
        # Check if they're consecutive or nearly consecutive
        ascii_diff = ord(first_letter) - ord(last_letter)
        return 1 <= ascii_diff <= 3  # Allow small gaps
    
    def _extract_alphabetical(self, row):
        """Extract alphabetical sequences from a row."""
        alpha_sequences = []
        
        for cell in row:
            # Find alphabetical sequences (at least 2 letters)
            matches = re.findall(r'[A-Za-z]{2,}', cell)
            alpha_sequences.extend(matches)
            
        return alpha_sequences
    
    def _extract_numbers(self, row):
        """Extract numbers from a row of cells."""
        numbers = []
        for cell in row:
            # Extract numerical values
            matches = re.findall(r'[-+]?\d+(?:\.\d+)?', cell)
            numbers.extend([float(match) for match in matches])
        return numbers
    
    def _merge_continuation_tables(self):
        """Process and merge the identified continuation tables."""
        # Build continuation chain
        continuation_map = {}
        for parent, child in self.continuation_tables:
            continuation_map[child.table_id] = parent
        
        # Find root tables (those that are not continuations themselves)
        processed_tables = set()
        
        for parent, child in self.continuation_tables:
            # Skip if already processed
            if child.table_id in processed_tables:
                continue
                
            # Find the root table by traversing up the chain
            root = parent
            while root.table_id in continuation_map:
                root = continuation_map[root.table_id]
            
            # Merge the child into the root table
            root.merge_continuation(child)
            processed_tables.add(child.table_id)
            
        # Clear the list after processing
        self.continuation_tables = []
    
    def _detect_nested_tables(self, tables: List[TableData]):
        """Detect nested tables with improved geometric and content analysis."""
        # Group tables by page for efficiency
        by_page = defaultdict(list)
        for table in tables:
            if not table.is_continuation:  # Skip continuations
                by_page[table.page_number].append(table)
        
        for page_num, page_tables in by_page.items():
            # First sort by area (largest first) to find parent tables
            page_tables.sort(key=lambda t: -(t.bbox[2]-t.bbox[0])*(t.bbox[3]-t.bbox[1]))
            
            # Build containment graph - a directed graph where edges represent containment
            containment_graph = defaultdict(list)
            
            for i, parent_candidate in enumerate(page_tables):
                for j, child_candidate in enumerate(page_tables[i+1:], i+1):
                    # Check geometric containment with improved precision
                    intersection = self._intersection_area(parent_candidate.bbox, child_candidate.bbox)
                    child_area = self._area(child_candidate.bbox)
                    
                    # Child must be mostly contained within parent
                    containment_ratio = intersection / child_area if child_area > 0 else 0
                    
                    if containment_ratio >= self.nested_table_threshold:
                        containment_graph[parent_candidate.table_id].append(
                            (child_candidate.table_id, containment_ratio)
                        )
            
            # Process the containment graph to establish proper nesting hierarchy
            self._establish_nesting_hierarchy(containment_graph, page_tables)
    
    def _establish_nesting_hierarchy(self, graph, tables):
        """Establish proper nesting hierarchy from containment graph."""
        # Get mapping from table_id to table object
        id_to_table = {t.table_id: t for t in tables}
        
        # Process each potential parent
        for parent_id, children in graph.items():
            parent = id_to_table[parent_id]
            
            # Sort children by containment ratio (highest first)
            children.sort(key=lambda x: -x[1])
            
            for child_id, ratio in children:
                child = id_to_table[child_id]
                
                # Check if child already has a better parent
                if child.parent_id:
                    current_parent = id_to_table[child.parent_id]
                    current_ratio = self._intersection_area(current_parent.bbox, child.bbox) / self._area(child.bbox)
                    
                    # Skip if current parent is better
                    if current_ratio > ratio:
                        continue
                
                # Establish parent-child relationship
                parent.has_nested_tables = True
                parent.nested_tables.append(child_id)
                child.parent_id = parent_id
                
                logger.info(f"Established nesting: {child_id} inside {parent_id} with ratio {ratio:.2f}")

    # ─── utility methods ───────────────────────────────────────────
    def _get_table_signature(self, table: TableData) -> str:
        """Generate a signature for a table based on content."""
        if not table.content or len(table.content) == 0:
            return ""
        
        # Use first row (header) and first column as signature
        header = table.get_header_signature()
        
        # Add first column content (skip header row)
        first_col = ""
        if len(table.content) > 1:
            first_col_cells = [row[0].strip().lower() for row in table.content[1:] 
                              if row and len(row) > 0]
            first_col = "∥".join(first_col_cells)
        
        return f"{header}◊{first_col}"
    
    def _fuzzy_cell_match(self, cell1: str, cell2: str) -> bool:
        """Check if two cells are similar despite minor differences."""
        # Normalize text
        c1 = re.sub(r'\s+', ' ', cell1.lower()).strip()
        c2 = re.sub(r'\s+', ' ', cell2.lower()).strip()
        
        # Empty cells match
        if not c1 and not c2:
            return True
            
        # If one is empty, no match
        if not c1 or not c2:
            return False
            
        # Check if one is substring of the other
        if c1 in c2 or c2 in c1:
            return True
            
        # Check for high character-level similarity
        from difflib import SequenceMatcher
        char_sim = SequenceMatcher(None, c1, c2).ratio()
        if char_sim >= 0.8:
            return True
            
        # Check for numerical similarity
        num1 = self._extract_numeric_value(c1)
        num2 = self._extract_numeric_value(c2)
        if num1 is not None and num2 is not None:
            # Small relative difference
            if max(num1, num2) > 0 and abs(num1 - num2) / max(num1, num2) < 0.05:
                return True
        
        return False
    
    def _extract_numeric_value(self, text):
        """Extract numeric value from text."""
        # Remove non-numeric chars except decimal point
        numeric_chars = re.findall(r'[-+]?\d*\.\d+|\d+', text)
        
        if not numeric_chars:
            return None
            
        try:
            return float(numeric_chars[0])
        except ValueError:
            return None
    
    def _intersection_area(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        """Calculate intersection area between two bounding boxes."""
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        return max(0, x1 - x0) * max(0, y1 - y0)
    
    def _area(self, bbox: Tuple[float, float, float, float]) -> float:
        """Calculate area of a bounding box."""
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    # ─── text extraction ──────────────────────────────────────────
    def _extract_text_outside(self, page, table_bboxes):
        """Extract text elements that don't overlap with tables."""
        elements = []
        for ln in page.extract_text_lines():
            text_bbox = (ln["x0"], ln["top"], ln["x1"], ln["bottom"])
            
            # Check if text overlaps with any table
            if not any(self._intersection_area(text_bbox, tb) > 0 for tb in table_bboxes):
                elements.append({
                    "type": "text",
                    "y": ln["top"],
                    "content": ln["text"],
                    "bbox": text_bbox  # Adding bbox for text elements
                })
        
        return elements
