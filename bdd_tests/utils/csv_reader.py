# utils/csv_reader.py

import csv
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

def load_csv_data(file_name: str) -> List[Dict]:
    """
    Reads data from a CSV file located in the 'data/' directory.

    Args:
        file_name (str): The name of the CSV file (e.g., 'complaint_data.csv').

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a row.
    
    Raises:
        FileNotFoundError: If the specified CSV file cannot be found.
    """
    # Construct the full path to the CSV file within the 'data' directory
    data_path = Path(__file__).parent.parent / "data" / file_name
    
    if not data_path.exists():
        logger.error(f"CSV data file not found at path: {data_path}")
        raise FileNotFoundError(f"Could not find {file_name} in the data directory.")
        
    logger.info(f"Loading test data from: {data_path}")
    
    records = []
    try:
        with open(data_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                records.append(row)
        logger.info(f"Successfully loaded {len(records)} records from {file_name}")
        return records
    except Exception as e:
        logger.error(f"Failed to read or parse CSV file {data_path}. Error: {e}")
        raise

