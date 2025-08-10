import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from .logger_config import get_logger


class DataLoader:
    """Centralized utility for loading test data from various sources."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize data loader with configurable data directory."""
        if data_dir is None:
            # Default to 'data' directory relative to project root
            self.data_dir = Path(__file__).parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.resources_dir = Path(__file__).parent.parent / "resources"
        self.logger = get_logger(__name__)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.resources_dir.mkdir(exist_ok=True, parents=True)
    
    def load_csv(self, filename: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """Load data from CSV file with proper error handling."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"CSV file not found: {filepath}")
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        self.logger.info(f"Loading CSV data from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                reader = csv.DictReader(file)
                data = list(reader)
            
            self.logger.info(f"Loaded {len(data)} rows from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV file {filename}: {e}")
            raise
    
    def load_json(self, filename: str, from_resources: bool = True) -> Dict[str, Any]:
        """Load data from JSON file."""
        if from_resources:
            filepath = self.resources_dir / filename
        else:
            filepath = self.data_dir / filename
        
        if not filepath.exists():
            self.logger.error(f"JSON file not found: {filepath}")
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        self.logger.info(f"Loading JSON data from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            self.logger.info(f"Loaded JSON data from {filename}")
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in file {filename}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load JSON file {filename}: {e}")
            raise
    
    def get_test_data_by_id(self, test_case_id: str, csv_file: str = "complaint_data.csv") -> Optional[Dict[str, Any]]:
        """Get specific test data row by test case ID."""
        data = self.load_csv(csv_file)
        for row in data:
            if row.get('test_case_id') == test_case_id:
                return row
        return None
    
    def save_results(
        self, 
        data: Any, 
        filename: str, 
        format: str = 'json'
    ) -> Path:
        """Save results to file."""
        reports_dir = Path(__file__).parent.parent / "reports"
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        filepath = reports_dir / filename
        
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2, default=str)
            
            elif format == 'csv' and isinstance(data, list):
                if data and isinstance(data[0], dict):
                    keys = data[0].keys()
                    with open(filepath, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(data)
            
            self.logger.info(f"Saved results to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {filename}: {e}")
            raise