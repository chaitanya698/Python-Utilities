import os
import sys
import pytest
from pytest_bdd import given
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import get_config, cleanup_config
from config.settings import Settings
from database.db_manager import DatabaseManager
from fixtures.api_service import ChatbotAPIClient
from fixtures.db_utils import DBUtils
from utils.data_loader import DataLoader
from utils.logger_config import get_logger, LoggerSetup
from utils.report_generator import BusinessReportGenerator

# Global test tracking
_report_generator = BusinessReportGenerator()
_test_results = []
_test_start_time = datetime.now()
_current_test_id = None


def pytest_addoption(parser):
    """Add command-line options."""
    parser.addoption("--env", action="store", default="qa", 
                     choices=["dev", "qa", "staging", "production"])
    parser.addoption("--report", action="store_true", default=True)
    parser.addoption("--report-title", default="BDD Test Automation Report")


def pytest_configure(config):
    """Configure test environment."""
    env = config.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    LoggerSetup.setup(log_level=os.getenv("LOG_LEVEL", "INFO"))
    logger = get_logger(__name__)
    logger.info(f"Starting test run in {env.upper()} environment")


# ===== Enhanced Data Loader =====
class RobustDataLoader(DataLoader):
    """Enhanced data loader with robust CSV handling."""
    
    def load_csv_with_encoding_fix(self, filename: str) -> List[Dict[str, Any]]:
        """Load CSV with automatic encoding detection and BOM handling."""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        logger = get_logger(__name__)
        
        # Try multiple encodings
        encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                import csv
                with open(filepath, 'r', encoding=encoding, newline='') as file:
                    reader = csv.DictReader(file)
                    data = list(reader)
                
                # Clean the data
                cleaned_data = []
                for row in data:
                    cleaned_row = {}
                    for key, value in row.items():
                        # Remove BOM and clean key
                        clean_key = key.replace('\ufeff', '').strip()
                        # Clean value
                        if isinstance(value, str):
                            clean_value = value.replace('\ufeff', '').strip()
                        else:
                            clean_value = value
                        cleaned_row[clean_key] = clean_value
                    cleaned_data.append(cleaned_row)
                
                logger.info(f"Successfully loaded {len(cleaned_data)} rows using {encoding}")
                return cleaned_data
                
            except (UnicodeDecodeError, UnicodeError) as e:
                logger.debug(f"Failed with {encoding}: {e}")
                continue
        
        raise Exception(f"Failed to load CSV with any encoding")
    
    def validate_test_data(self, row: Dict[str, Any]) -> bool:
        """Validate if a test data row has valid data."""
        # Check for test_case_id
        test_case_id = row.get('test_case_id', '').strip()
        if not test_case_id:
            return False
        
        # Check if at least one chatText field has valid data
        for i in range(1, 15):  # Support up to chatText14
            chat_text = row.get(f'chatText{i}', '').strip()
            if chat_text and chat_text.lower() not in ['', 'n/a', 'null', 'none']:
                return True
        
        return False


# ===== Session Fixtures =====
@pytest.fixture(scope="session")
def config(pytestconfig) -> Settings:
    """Load configuration."""
    env = pytestconfig.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    return get_config()


@pytest.fixture(scope="session")
def db_manager(config: Settings):
    """Create database manager."""
    logger = get_logger(__name__)
    try:
        manager = DatabaseManager(config)
        yield manager
    except Exception as e:
        logger.error(f"DB initialization failed: {e}")
        pytest.skip(f"Database unavailable: {e}")
    finally:
        if 'manager' in locals():
            manager.close()


@pytest.fixture(scope="session")
def api_client(config: Settings):
    """Create API client."""
    from utils.request_response_tracker import RequestResponseTracker
    tracker = RequestResponseTracker()
    client = ChatbotAPIClient(config, tracker)
    yield client
    client.close()


@pytest.fixture(scope="session")
def db_utils(db_manager: DatabaseManager):
    """Provide database utilities."""
    return DBUtils(db_manager)


@pytest.fixture(scope="session")
def robust_data_loader():
    """Provide enhanced data loader."""
    return RobustDataLoader()


# ===== Test Context Fixtures =====
@pytest.fixture
def test_context():
    """Provide clean test context."""
    return {
        'test_id': None,
        'test_data': {},
        'csv_data': {},
        'response': None,
        'conversation_id': None,
        'step_results': [],
        'available_chat_texts': {}
    }


@pytest.fixture
def given_api_is_available(api_client: ChatbotAPIClient, test_context: Dict[str, Any]):
    """Ensure API is available."""
    test_context['api_client'] = api_client
    return test_context


@pytest.fixture
def given_test_data_loaded(robust_data_loader: RobustDataLoader, test_context: Dict[str, Any], test_data_row: Dict[str, Any]):
    """Load and prepare test data."""
    logger = get_logger(__name__)
    
    # Store raw CSV data
    test_context['csv_data'] = test_data_row
    test_context['test_case_id'] = test_data_row.get('test_case_id', 'unknown')
    
    # Identify available chatText fields
    available_chat_texts = {}
    for i in range(1, 15):  # Support up to chatText14
        field_name = f'chatText{i}'
        value = test_data_row.get(field_name, '').strip()
        
        # Only include non-empty, valid values
        if value and value.lower() not in ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty']:
            available_chat_texts[field_name] = value
            logger.debug(f"Found valid {field_name}: {value[:50]}...")
    
    test_context['available_chat_texts'] = available_chat_texts
    test_context['test_data'] = test_data_row
    
    logger.info(f"Test case {test_context['test_case_id']} has {len(available_chat_texts)} valid chatText fields")
    
    return test_context


# ===== Test Generation =====
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV."""
    if "test_data_row" in metafunc.fixturenames:
        logger = get_logger(__name__)
        data_loader = RobustDataLoader()
        
        try:
            # Load CSV with robust encoding handling
            csv_file = "complaint_data.csv"
            logger.info(f"Loading test data from {csv_file}")
            
            all_data = data_loader.load_csv_with_encoding_fix(csv_file)
            logger.info(f"Loaded {len(all_data)} total rows from CSV")
            
            # Filter valid test cases
            valid_data = []
            test_ids = []
            
            for row in all_data:
                if data_loader.validate_test_data(row):
                    valid_data.append(row)
                    test_ids.append(row.get('test_case_id', f'TC_{len(valid_data):03d}'))
                else:
                    test_case_id = row.get('test_case_id', 'unknown')
                    logger.debug(f"Skipping invalid test case: {test_case_id}")
            
            if valid_data:
                logger.info(f"Generated {len(valid_data)} valid test cases")
                metafunc.parametrize("test_data_row", valid_data, ids=test_ids)
            else:
                logger.error("No valid test data found")
                # Create dummy test to report the issue
                dummy_data = [{"test_case_id": "NO_VALID_DATA", "error": "No valid test data in CSV"}]
                metafunc.parametrize("test_data_row", dummy_data, ids=["NO_VALID_DATA"])
                
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            error_data = [{"test_case_id": "DATA_LOAD_ERROR", "error": str(e)}]
            metafunc.parametrize("test_data_row", error_data, ids=["DATA_LOAD_ERROR"])


# ===== BDD Hooks for Reporting =====
@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_step(request, feature, scenario, step, step_func):
    """Before step execution."""
    global _current_test_id
    if not hasattr(request.node, 'step_results'):
        request.node.step_results = []


@pytest.hookimpl(tryfirst=True)
def pytest_bdd_after_step(request, feature, scenario, step, step_func, step_func_args):
    """After successful step execution."""
    global _current_test_id
    details = dict(step_func_args) if step_func_args else {}
    test_context = details.get('test_context', {})
    
    step_info = {
        'test_id': _current_test_id or test_context.get('test_case_id', 'unknown'),
        'type': step.keyword.strip().lower(),
        'description': step.name,
        'status': 'passed',
        'request': test_context.get('request'),
        'response': test_context.get('response')
    }
    _report_generator.add_step_result(step_info)


@pytest.hookimpl(tryfirst=True)
def pytest_bdd_step_error(request, feature, scenario, step, step_func, step_func_args, exception):
    """After step failure."""
    global _current_test_id
    logger = get_logger(__name__)
    
    details = dict(step_func_args) if step_func_args else {}
    test_context = details.get('test_context', {})
    
    step_info = {
        'test_id': _current_test_id or test_context.get('test_case_id', 'unknown'),
        'type': step.keyword.strip().lower(),
        'description': step.name,
        'status': 'failed',
        'error': str(exception),
        'request': test_context.get('request'),
        'response': test_context.get('response')
    }
    _report_generator.add_step_result(step_info)
    logger.error(f"Step failed: {step.name} - {exception}")


def pytest_sessionfinish(session, exitstatus):
    """Generate report after tests complete."""
    logger = get_logger(__name__)
    
    if _test_results and session.config.getoption("--report"):
        try:
            metadata = {
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
                'total_duration': (datetime.now() - _test_start_time).total_seconds(),
                'exit_status': exitstatus
            }
            
            report_path = _report_generator.generate_report(_test_results, metadata)
            logger.info(f"Report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    # Print summary
    total = len(_test_results)
    passed = sum(1 for r in _test_results if r.get('status') == 'passed')
    failed = sum(1 for r in _test_results if r.get('status') == 'failed')
    
    logger.info(f"Test Summary: Total={total}, Passed={passed}, Failed={failed}")
