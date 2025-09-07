# Add these enhanced fixtures to your existing conftest.py

import pytest
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path

# Enhanced Data Loader for CSV with proper encoding
class EnhancedComplaintDataLoader:
    """Enhanced data loader specifically for complaint capture data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path("bdd_tests/data")
    
    def load_complaint_capture_csv(self, filename: str = "complaint_capture_data.csv") -> List[Dict[str, Any]]:
        """Load complaint capture CSV with all fields."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            # Try alternative locations
            alternative_paths = [
                Path("data") / filename,
                Path.cwd() / filename,
                Path(__file__).parent / "data" / filename,
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    filepath = alt_path
                    break
            else:
                raise FileNotFoundError(f"CSV file not found: {filename}")
        
        import csv
        import codecs
        
        # Detect and handle BOM
        with open(filepath, 'rb') as f:
            raw = f.read()
            if raw.startswith(codecs.BOM_UTF8):
                raw = raw[len(codecs.BOM_UTF8):]
        
        # Parse CSV
        text = raw.decode('utf-8-sig', errors='ignore')
        lines = text.strip().split('\n')
        
        if not lines:
            return []
        
        # Use csv.DictReader for proper parsing
        import io
        csv_io = io.StringIO(text)
        reader = csv.DictReader(csv_io)
        
        data = []
        for row in reader:
            # Clean all values
            cleaned_row = {}
            for key, value in row.items():
                # Remove BOM and clean
                clean_key = key.replace('\ufeff', '').strip() if key else ''
                clean_value = value.replace('\ufeff', '').strip() if value else ''
                
                if clean_key:
                    cleaned_row[clean_key] = clean_value
            
            # Only add rows with test_case_id
            if cleaned_row.get('test_case_id'):
                data.append(cleaned_row)
                self.logger.debug(f"Loaded test case: {cleaned_row['test_case_id']}")
        
        self.logger.info(f"Loaded {len(data)} test cases from {filename}")
        return data
    
    def validate_test_case(self, row: Dict[str, Any]) -> bool:
        """Validate if a test case has sufficient data."""
        # Must have test_case_id
        if not row.get('test_case_id', '').strip():
            return False
        
        # Check if at least one workflow field has data
        workflow_fields = [
            'complaint_date', 'complaint_method', 'Full Name', 'account_number',
            'complaint_eloboration', 'follow_up_question1', 'followup_question_2',
            'clarification_revise_action', 'clarification_revise_1',
            'clarification_revise_2', 'clarification_revise_3', 'clarification_revise_4',
            'unauthorized_account_handling', 'contact_willingness_response',
            'Add_new_phone_email', 'complaint_submission_action'
        ]
        
        for field in workflow_fields:
            value = row.get(field, '').strip()
            if value and value.lower() not in ['', 'n/a', 'null', 'none', 'nan']:
                return True
        
        return False


@pytest.fixture(scope="session")
def enhanced_data_loader():
    """Provide enhanced complaint data loader."""
    return EnhancedComplaintDataLoader()


@pytest.fixture
def test_data_row_enhanced(request):
    """Enhanced fixture to provide test data row."""
    # This will be parameterized by pytest_generate_tests
    return request.param


@pytest.fixture
def given_test_data_loaded_enhanced(enhanced_data_loader, test_context, test_data_row_enhanced):
    """Enhanced fixture to load and prepare test data."""
    logger = logging.getLogger(__name__)
    
    # Store complete CSV data
    test_context['csv_data'] = test_data_row_enhanced
    test_context['test_case_id'] = test_data_row_enhanced.get('test_case_id', 'unknown')
    
    # Identify available workflow fields
    available_fields = {}
    workflow_fields = [
        'complaint_date', 'complaint_method', 'Full Name', 'account_number',
        'complaint_eloboration', 'follow_up_question1', 'followup_question_2',
        'clarification_revise_action', 'clarification_revise_1',
        'clarification_revise_2', 'clarification_revise_3', 'clarification_revise_4',
        'unauthorized_account_handling', 'contact_willingness_response',
        'Add_new_phone_email', 'complaint_submission_action'
    ]
    
    for field in workflow_fields:
        value = test_data_row_enhanced.get(field, '').strip()
        if value and value.lower() not in ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty']:
            available_fields[field] = value
            logger.debug(f"Found valid {field}: {value[:50]}...")
    
    test_context['available_workflow_fields'] = available_fields
    test_context['test_data'] = test_data_row_enhanced
    
    # Also store expected values
    expected_fields = {}
    for field in workflow_fields:
        expected_key = f"expected_{field}"
        expected_value = test_data_row_enhanced.get(expected_key, '').strip()
        if expected_value and expected_value.lower() not in ['', 'n/a', 'null', 'none', 'nan']:
            expected_fields[expected_key] = expected_value
    
    test_context['expected_fields'] = expected_fields
    
    logger.info(f"Test case {test_context['test_case_id']} loaded:")
    logger.info(f"  - Available workflow fields: {len(available_fields)}")
    logger.info(f"  - Expected response fields: {len(expected_fields)}")
    logger.info(f"  - Error scenario: {test_data_row_enhanced.get('initial_request_error_key', 'none')}")
    
    return test_context


# Enhanced test generation function
def pytest_generate_tests_enhanced(metafunc):
    """Enhanced test generation from complaint_capture_data.csv."""
    if "test_data_row_enhanced" in metafunc.fixturenames:
        logger = logging.getLogger(__name__)
        data_loader = EnhancedComplaintDataLoader()
        
        try:
            # Load CSV data
            csv_file = "complaint_capture_data.csv"
            logger.info(f"Loading test data from {csv_file}")
            
            all_data = data_loader.load_complaint_capture_csv(csv_file)
            logger.info(f"Loaded {len(all_data)} total rows from CSV")
            
            # Filter valid test cases
            valid_data = []
            test_ids = []
            
            for row in all_data:
                if data_loader.validate_test_case(row):
                    valid_data.append(row)
                    test_ids.append(row.get('test_case_id', f'TC_{len(valid_data):03d}'))
                else:
                    test_case_id = row.get('test_case_id', 'unknown')
                    logger.debug(f"Skipping invalid test case: {test_case_id}")
            
            if valid_data:
                logger.info(f"Generated {len(valid_data)} valid test cases")
                metafunc.parametrize("test_data_row_enhanced", valid_data, ids=test_ids)
            else:
                logger.error("No valid test data found")
                # Create dummy test to report the issue
                dummy_data = [{"test_case_id": "NO_VALID_DATA", "error": "No valid test data in CSV"}]
                metafunc.parametrize("test_data_row_enhanced", dummy_data, ids=["NO_VALID_DATA"])
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            error_data = [{"test_case_id": "DATA_LOAD_ERROR", "error": str(e)}]
            metafunc.parametrize("test_data_row_enhanced", error_data, ids=["DATA_LOAD_ERROR"])


# Update the existing given_test_data_loaded fixture to use enhanced version
@pytest.fixture
def given_test_data_loaded(enhanced_data_loader, test_context, test_data_row):
    """Redirect to enhanced fixture."""
    # This maintains backward compatibility
    return given_test_data_loaded_enhanced(enhanced_data_loader, test_context, test_data_row)
