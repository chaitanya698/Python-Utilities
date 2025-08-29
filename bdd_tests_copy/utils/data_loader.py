import csv
import json
import codecs
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .logger_config import get_logger


class DataLoader:
    """Centralized utility for loading test data with robust encoding support."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader."""
        if data_dir is None:
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.resources_dir = Path(__file__).parent.parent / "resources"
        self.logger = get_logger(__name__)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.resources_dir.mkdir(exist_ok=True, parents=True)
    
    def detect_encoding(self, filepath: Path) -> str:
        """Detect file encoding by trying to read with different encodings."""
        encodings = [
            'utf-8-sig',  # UTF-8 with BOM
            'utf-8',      # UTF-8 without BOM
            'utf-16',     # UTF-16
            'latin1',     # Latin-1
            'cp1252',     # Windows-1252
            'iso-8859-1', # ISO-8859-1
        ]
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try reading first 1KB
                self.logger.debug(f"Detected encoding: {encoding}")
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        # Default to utf-8 if nothing works
        self.logger.warning(f"Could not detect encoding, defaulting to utf-8")
        return 'utf-8'
    
    def clean_csv_value(self, value: Any) -> str:
        """Clean CSV value by removing BOM, quotes, and extra whitespace."""
        if value is None:
            return ''
        
        # Convert to string
        text = str(value)
        
        # Remove BOM characters
        text = text.replace('\ufeff', '')  # UTF-8 BOM
        text = text.replace('\ufffe', '')  # UTF-16 BOM
        text = text.replace('\u0000', '')  # Null characters
        
        # Strip whitespace
        text = text.strip()
        
        # Remove surrounding quotes if present
        if len(text) >= 2:
            if (text.startswith('"') and text.endswith('"')) or \
               (text.startswith("'") and text.endswith("'")):
                text = text[1:-1].strip()
        
        return text
    
    def load_csv(self, filename: str, encoding: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load CSV with automatic encoding detection and cleaning."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"CSV file not found: {filepath}")
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        # Detect encoding if not provided
        if encoding is None:
            encoding = self.detect_encoding(filepath)
        
        self.logger.info(f"Loading CSV from {filepath} with encoding {encoding}")
        
        try:
            data = []
            
            with open(filepath, 'r', encoding=encoding, newline='') as file:
                # Skip BOM if present
                if encoding == 'utf-8-sig' or encoding == 'utf-8':
                    first_char = file.read(1)
                    if first_char != '\ufeff':
                        file.seek(0)  # Reset if no BOM
                
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, 1):
                    # Clean all keys and values
                    cleaned_row = {}
                    for key, value in row.items():
                        clean_key = self.clean_csv_value(key)
                        clean_value = self.clean_csv_value(value)
                        
                        # Skip empty keys
                        if clean_key:
                            cleaned_row[clean_key] = clean_value
                    
                    # Only add non-empty rows
                    if cleaned_row:
                        data.append(cleaned_row)
                    else:
                        self.logger.debug(f"Skipping empty row {row_num}")
            
            self.logger.info(f"Successfully loaded {len(data)} rows from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV {filename}: {e}")
            raise
    
    def load_complaint_data_csv(self, filename: str = "complaint_data.csv") -> List[Dict[str, Any]]:
        """Load complaint data CSV with validation."""
        data = self.load_csv(filename)
        
        # Validate and filter data
        valid_data = []
        for row in data:
            test_case_id = row.get('test_case_id', '').strip()
            
            if not test_case_id:
                self.logger.debug("Skipping row without test_case_id")
                continue
            
            # Check if at least one chatText field has valid data
            has_valid_chat = False
            for i in range(1, 15):  # Support up to chatText14
                chat_text = row.get(f'chatText{i}', '').strip()
                if self.is_valid_value(chat_text):
                    has_valid_chat = True
                    break
            
            if has_valid_chat:
                valid_data.append(row)
                self.logger.debug(f"Added test case: {test_case_id}")
            else:
                self.logger.debug(f"Skipping test case {test_case_id} - no valid chat text")
        
        self.logger.info(f"Validated {len(valid_data)} test cases from {len(data)} total rows")
        return valid_data
    
    def is_valid_value(self, value: str) -> bool:
        """Check if a value is valid (not empty or placeholder)."""
        if not value:
            return False
        
        invalid_values = [
            '', 'n/a', 'na', 'null', 'none', 'nan', 
            '""', "''", 'empty', '-', 'nil', 'undefined'
        ]
        
        return value.lower() not in invalid_values
    
    def load_json(self, filename: str, from_resources: bool = True) -> Dict[str, Any]:
        """Load JSON file."""
        if from_resources:
            filepath = self.resources_dir / filename
        else:
            filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"JSON file not found: {filepath}")
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        self.logger.info(f"Loading JSON from {filepath}")
        
        try:
            # Try to detect and handle BOM in JSON files too
            with open(filepath, 'rb') as file:
                raw_data = file.read()
                
                # Remove BOM if present
                if raw_data.startswith(codecs.BOM_UTF8):
                    raw_data = raw_data[len(codecs.BOM_UTF8):]
                
                # Decode and parse
                text_data = raw_data.decode('utf-8')
                data = json.loads(text_data)
            
            self.logger.info(f"Successfully loaded JSON from {filename}")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {filename}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load JSON {filename}: {e}")
            raise
    
    def get_test_data_by_id(self, test_case_id: str, csv_file: str = "complaint_data.csv") -> Optional[Dict[str, Any]]:
        """Get specific test data by test case ID."""
        data = self.load_csv(csv_file)
        
        for row in data:
            if self.clean_csv_value(row.get('test_case_id', '')) == test_case_id:
                return row
        
        self.logger.warning(f"Test case {test_case_id} not found in {csv_file}")
        return None
    
    def get_available_chat_texts(self, row: Dict[str, Any]) -> Dict[str, str]:
        """Extract available chat text fields from a data row."""
        available = {}
        
        for i in range(1, 15):  # Support up to chatText14
            field_name = f'chatText{i}'
            value = self.clean_csv_value(row.get(field_name, ''))
            
            if self.is_valid_value(value):
                available[field_name] = value
        
        return available
    
    def save_results(self, data: Any, filename: str, format: str = 'json') -> Path:
        """Save results to file."""
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        filepath = reports_dir / filename
        
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2, default=str, ensure_ascii=False)
            
            elif format == 'csv' and isinstance(data, list):
                if data and isinstance(data[0], dict):
                    keys = data[0].keys()
                    with open(filepath, 'w', newline='', encoding='utf-8-sig') as file:
                        writer = csv.DictWriter(file, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(data)
            
            self.logger.info(f"Saved results to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise


class RobustDataLoader(DataLoader):
    """Enhanced data loader with additional robustness features."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize robust data loader."""
        super().__init__(data_dir)
        self.logger = get_logger(f"{__name__}.RobustDataLoader")
    
    def load_csv_with_fallback(self, filename: str) -> List[Dict[str, Any]]:
        """Load CSV with multiple fallback strategies."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Try alternative locations
            alternative_paths = [
                Path.cwd() / "data" / filename,
                Path.cwd() / filename,
                Path(__file__).parent.parent.parent / "data" / filename,
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    self.logger.info(f"Found file at alternative location: {alt_path}")
                    filepath = alt_path
                    break
            else:
                raise FileNotFoundError(f"CSV file not found in any location: {filename}")
        
        # Try loading with parent method
        try:
            return self.load_csv(filename)
        except Exception as e:
            self.logger.error(f"Standard loading failed: {e}")
            
            # Try manual parsing as fallback
            return self._manual_csv_parse(filepath)
    
    def _manual_csv_parse(self, filepath: Path) -> List[Dict[str, Any]]:
        """Manually parse CSV with maximum compatibility."""
        self.logger.info("Attempting manual CSV parsing")
        
        # Read raw content
        with open(filepath, 'rb') as f:
            raw_content = f.read()
        
        # Remove BOM
        if raw_content.startswith(codecs.BOM_UTF8):
            raw_content = raw_content[len(codecs.BOM_UTF8):]
        
        # Decode with fallback
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                text_content = raw_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text_content = raw_content.decode('utf-8', errors='ignore')
        
        # Parse CSV manually
        lines = text_content.strip().split('\n')
        if not lines:
            return []
        
        # Get headers
        headers = [self.clean_csv_value(h) for h in lines[0].split(',')]
        
        # Parse rows
        data = []
        for line in lines[1:]:
            if not line.strip():
                continue
            
            values = line.split(',')
            row = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    row[header] = self.clean_csv_value(values[i])
                else:
                    row[header] = ''
            
            data.append(row)
        
        self.logger.info(f"Manual parsing succeeded: {len(data)} rows")
        return data
