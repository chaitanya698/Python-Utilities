import csv
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from core.utils.logger import get_logger

class DataLoader:
    """Utility for loading test data from various sources."""

    def __init__(self, data_dir: Path = Path("tests/data")):
        self.data_dir = data_dir
        self.logger = get_logger(__name__)

    def load_csv(self, filename: str, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """Load data from CSV file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        self.logger.info(f"Loading CSV data from: {filepath}")
        
        try:
            records = []
            with open(filepath, mode='r', encoding=encoding) as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Clean up whitespace
                    cleaned_row = {k.strip(): v.strip() if v else v for k, v in row.items()}
                    records.append(cleaned_row)
            
            self.logger.info(f"Loaded {len(records)} records from {filename}")
            return records
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV file: {e}")
            raise

    def load_json(self, filename: str) -> Dict[str, Any]:
        """Load data from JSON file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        self.logger.info(f"Loading JSON data from: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            self.logger.info(f"Loaded JSON data from {filename}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {e}")
            raise

    def load_excel(self, filename: str, sheet_name: str = None) -> pd.DataFrame:
        """Load data from Excel file."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Excel file not found: {filepath}")
        
        self.logger.info(f"Loading Excel data from: {filepath}")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            self.logger.info(f"Loaded {len(df)} rows from {filename}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load Excel file: {e}")
            raise

    def save_results(self, data: Any, filename: str, format: str = 'json'):
        """Save results to file."""
        filepath = self.data_dir / filename
        
        try:
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as file:
                    json.dump(data, file, indent=2)
            elif format == 'csv' and isinstance(data, list):
                if data and isinstance(data[0], dict):
                    keys = data[0].keys()
                    with open(filepath, 'w', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(data)
            
            self.logger.info(f"Saved results to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        raise
