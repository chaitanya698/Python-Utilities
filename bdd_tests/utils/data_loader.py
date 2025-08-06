import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Union
import pandas as pd

from .logger_config import get_logger


class DataLoader:
    """Utility for loading test data from various sources."""
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.logger = get_logger(__name__)
        
        if not self.data_dir.exists():
            self.logger.warning(f"Data directory does not exist: {self.data_dir}")
    
    def load_csv_data(file_name: str) -> list:
        # Construct the full path to the data file
        data_path = Path(__file__).parent.parent / 'data' / file_name
        logger.info(f"Attempting to load data from: {data_path}")
    
        if not data_path.exists():
            logger.error(f"Data file not found at: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
        try:
            with open(data_path, mode='r', encoding='utf-8') as infile:
                # Use DictReader to make it easy to access columns by name
                reader = csv.DictReader(infile)
                data = [row for row in reader]
                logger.info(f"Successfully loaded {len(data)} rows from {file_name}.")
                return data
        except Exception as e:
            logger.error(f"An error occurred while reading the CSV file {file_name}: {e}")
            raise
    
    def load_json(self, filename: str, dir_path: str = "resources") -> Dict[str, Any]:
        """Load data from JSON file."""
        filepath = Path(dir_path) / filename
        
        if not filepath.exists():
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
    
    def save_results(
        self, 
        data: Any, 
        filename: str, 
        format: str = 'json'
    ) -> Path:
        """Save results to file."""
        filepath = Path("reports") / filename
        filepath.parent.mkdir(exist_ok=True)
        
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
                else:
                    raise ValueError("Data must be a list of dictionaries for CSV format")
                    
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Saved results to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save results to {filename}: {e}")
            raise
