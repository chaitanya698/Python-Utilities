import os
import sys
import csv
import json
import platform
import atexit
import traceback
from datetime import datetime
from typing import Dict, Any, Generator, List, Optional
from pathlib import Path

import pytest
from pytest_bdd import given

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.loader import get_config, cleanup_config
from config.settings import Settings
from database.db_manager import DatabaseManager
from fixtures.api_service import ChatbotAPIClient
from fixtures.db_utils import DBUtils
from utils.data_loader import DataLoader
from utils.logger_config import get_logger, LoggerSetup
from utils.request_response_tracker import RequestResponseTracker
from utils.report_generator import BusinessReportGenerator

# Global instances
_request_response_tracker = RequestResponseTracker()
_report_generator = BusinessReportGenerator()
_test_results = []
_test_start_time = datetime.now()
_current_test_id = None


# ===== Command Line Options =====
def pytest_addoption(parser):
    """Add command-line options for runtime configuration."""
    parser.addoption(
        "--env",
        action="store",
        default="qa",
        choices=["dev", "qa", "staging", "production"],
        help="Environment to run tests against (default: qa)"
    )
    parser.addoption(
        "--report",
        action="store_true",
        default=True,
        help="Generate HTML report after test run"
    )
    parser.addoption(
        "--report-title",
        default="BDD Test Automation Report",
        help="Title for the HTML report"
    )


# ===== Pytest Configuration =====
def pytest_configure(config):
    """Configure test environment before test run."""
    env = config.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    LoggerSetup.setup(log_level=log_level)
    
    logger = get_logger(__name__)
    logger.info(f"=" * 60)
    logger.info(f"Starting test run in {env.upper()} environment")
    logger.info(f"=" * 60)
    
    # Register cleanup
    atexit.register(cleanup_config)


def pytest_html_report_title(report):
    """Set custom title for the HTML report."""
    report.title = "Complaint AI Chatbot - Test Report"


# ===== BDD Hooks =====
@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_scenario(request, feature, scenario):
    """Initialize a list to hold step results for the current scenario."""
    request.node.step_results = []


@pytest.hookimpl(tryfirst=True)
def pytest_bdd_before_step(request, feature, scenario, step, step_func):
    """Capture step execution start."""
    global _current_test_id
    if not hasattr(request.node, 'step_results'):
        request.node.step_results = []


@pytest.hookimpl(tryfirst=True)
def pytest_bdd_after_step(request, feature, scenario, step, step_func, step_func_args):
    """Capture passed step after execution, including request and response details."""
    global _current_test_id
    details = dict(step_func_args) if step_func_args else {}
    test_context = details.get('test_context', {})
    
    # --- MODIFICATION: Capture request and response for the report ---
    step_info = {
        'test_id': _current_test_id or 'unknown',
        'type': step.keyword.strip().lower(),
        'description': step.name,
        'status': 'passed',
        'request': test_context.get('request'),
        'response': test_context.get('response')
    }
    _report_generator.add_step_result(step_info)
    
    # Clear request/response from context to avoid carrying it to the next step
    test_context['request'] = None
    test_context['response'] = None


@pytest.hookimpl(tryfirst=True)
def pytest_bdd_step_error(request, feature, scenario, step, step_func, step_func_args, exception):
    """Capture failed step, including request and response details."""
    global _current_test_id
    details = dict(step_func_args) if step_func_args else {}
    test_context = details.get('test_context', {})
    
    error_msg = f"{type(exception).__name__}: {exception}"
    tb_str = ''.join(traceback.format_tb(exception.__traceback__))
    
    # --- MODIFICATION: Capture request and response for the report ---
    step_info = {
        'test_id': _current_test_id or 'unknown',
        'type': step.keyword.strip().lower(),
        'description': step.name,
        'status': 'failed',
        'error': f"{error_msg}\n{tb_str}",
        'request': test_context.get('request'),
        'response': test_context.get('response')
    }
    _report_generator.add_step_result(step_info)
    logger = get_logger(__name__)
    logger.error(f"Step '{step.name}' failed: {error_msg}")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    """Capture test results for reporting."""
    global _current_test_id
    outcome = yield
    report = outcome.get_result()
    
    if report.when == "setup":
        _current_test_id = item.nodeid.split('::')[-1]
    
    elif report.when == "call":
        test_result = {
            'test_id': item.nodeid.split('::')[-1],
            'description': getattr(item, 'description', item.name),
            'status': report.outcome,
            'duration': report.duration,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tags': [marker.name for marker in item.iter_markers()],
            'environment': os.getenv('ENVIRONMENT', 'unknown')
        }
        
        if hasattr(report, 'longrepr') and report.longrepr:
            test_result['error'] = str(report.longrepr)
        
        _test_results.append(test_result)


# ===== Session Fixtures =====
@pytest.fixture(scope="session")
def config(pytestconfig) -> Settings:
    """Load configuration once per session."""
    env = pytestconfig.getoption("--env")
    os.environ["ENVIRONMENT"] = env
    
    logger = get_logger(__name__)
    logger.info(f"Loading configuration for {env} environment")
    
    try:
        settings = get_config()
        logger.info("Configuration loaded successfully")
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        pytest.fail(f"Configuration loading failed: {e}")


@pytest.fixture(scope="session")
def request_response_tracker() -> RequestResponseTracker:
    """Provide request/response tracker for the session."""
    return _request_response_tracker


@pytest.fixture(scope="session")
def db_manager(config: Settings) -> Generator[DatabaseManager, None, None]:
    """Create database manager for the session."""
    logger = get_logger(__name__)
    logger.info("Initializing database manager")
    
    try:
        manager = DatabaseManager(config)
        logger.info("Database manager initialized successfully")
        yield manager
    except Exception as e:
        logger.error(f"Failed to initialize database manager: {e}")
        pytest.skip(f"Database initialization failed: {e}")
    finally:
        if 'manager' in locals():
            manager.close()
            logger.info("Database manager closed")


@pytest.fixture(scope="session")
def api_client(config: Settings, request_response_tracker: RequestResponseTracker) -> Generator[ChatbotAPIClient, None, None]:
    """Create API client for the session."""
    logger = get_logger(__name__)
    logger.info("Initializing API client")
    
    client = ChatbotAPIClient(config, request_response_tracker)
    yield client
    client.close()
    logger.info("API client closed")


@pytest.fixture(scope="session")
def db_utils(db_manager: DatabaseManager) -> DBUtils:
    """Provide database utilities."""
    return DBUtils(db_manager)


@pytest.fixture(scope="session")
def data_loader() -> DataLoader:
    """Create data loader instance."""
    return DataLoader()


class RobustDataLoader(DataLoader):
    """Enhanced data loader with robust encoding handling for CSV files."""
    
    def load_csv_with_multiple_encodings(self, filename: str, encodings: List[str] = None) -> List[Dict[str, Any]]:
        """Load CSV with multiple encoding attempts to handle BOM and various encodings."""
        if encodings is None:
            # Comprehensive list of encodings to try, prioritizing UTF variants that handle BOM
            encodings = [
                'utf-8-sig',      # UTF-8 with BOM
                'utf-8',          # Standard UTF-8
                'utf-16',         # UTF-16 with BOM detection
                'utf-16le',       # UTF-16 Little Endian
                'utf-16be',       # UTF-16 Big Endian  
                'latin1',         # ISO-8859-1
                'cp1252',         # Windows-1252
                'iso-8859-1',     # ISO Latin-1
                'ascii'           # ASCII fallback
            ]
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        
        logger = get_logger(__name__)
        logger.info(f"Attempting to load CSV: {filepath}")
        
        for encoding in encodings:
            try:
                logger.debug(f"Trying encoding: {encoding}")
                
                with open(filepath, 'r', encoding=encoding, newline='') as file:
                    # Test read a small sample first
                    file.seek(0)
                    sample = file.read(512)  # Read first 512 bytes
                    
                    # Reset to beginning and read full content
                    file.seek(0)
                    reader = csv.DictReader(file)
                    data = list(reader)
                
                logger.info(f"✅ Successfully loaded {len(data)} rows using {encoding} encoding")
                return data
                
            except (UnicodeDecodeError, UnicodeError) as e:
                logger.debug(f"❌ Encoding {encoding} failed: {e}")
                continue
            except Exception as e:
                logger.warning(f"⚠️  Failed with {encoding}: {type(e).__name__}: {e}")
                continue
        
        # If all encodings fail, provide detailed error message
        raise Exception(
            f"❌ Failed to load CSV file '{filename}' with any encoding. "
            f"Tried encodings: {', '.join(encodings)}. "
            f"File may be corrupted or in an unsupported format."
        )
    
    def clean_csv_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean CSV data by removing BOM characters and normalizing values."""
        cleaned_data = []
        logger = get_logger(__name__)
        
        for row_idx, row in enumerate(data):
            cleaned_row = {}
            
            for original_key, value in row.items():
                # Clean column names - remove BOM characters and whitespace
                clean_key = (original_key
                           .replace('\ufeff', '')      # UTF-8 BOM
                           .replace('\ufffe', '')      # UTF-16 BOM  
                           .replace('\u0000', '')      # Null characters
                           .strip())
                
                # Clean values
                if isinstance(value, str):
                    clean_value = (value
                                 .replace('\ufeff', '')
                                 .replace('\ufffe', '')
                                 .replace('\u0000', '')
                                 .strip())
                else:
                    clean_value = value
                
                cleaned_row[clean_key] = clean_value
            
            cleaned_data.append(cleaned_row)
            
            if row_idx == 0:
                logger.debug(f"Sample cleaned row keys: {list(cleaned_row.keys())}")
        
        return cleaned_data
    
    def load_complaint_workflow_csv(self, filename: str = "complaint_data.csv") -> List[Dict[str, Any]]:
        """Load and validate complaint workflow CSV data with robust encoding and validation."""
        logger = get_logger(__name__)
        logger.info(f"🔄 Loading complaint workflow CSV: {filename}")
        
        try:
            # Load with robust encoding handling
            raw_data = self.load_csv_with_multiple_encodings(filename)
            logger.info(f"📄 Raw data loaded: {len(raw_data)} rows")
            
            # Clean the data
            cleaned_data = self.clean_csv_data(raw_data)
            logger.info(f"🧹 Data cleaned: {len(cleaned_data)} rows")
            
            # Validate and filter data
            validated_data = []
            chat_text_columns = [f'chatText{i}' for i in range(1, 12)]
            
            for row in cleaned_data:
                test_case_id = row.get('test_case_id', '').strip()
                
                # Skip rows without test case ID
                if not test_case_id:
                    logger.debug("⚠️  Skipping row without test_case_id")
                    continue
                
                # Check for valid chatText data
                valid_chat_texts = []
                for col in chat_text_columns:
                    value = str(row.get(col, '')).strip()
                    # Consider non-empty values that aren't common placeholders
                    if value and value.lower() not in ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty', 'no', 'false']:
                        valid_chat_texts.append(col)
                
                if valid_chat_texts:
                    validated_data.append(row)
                    logger.debug(f"✅ {test_case_id}: {len(valid_chat_texts)} valid chatText columns")
                else:
                    logger.info(f"⚠️  {test_case_id}: No valid chatText data - skipping")
            
            logger.info(f"✅ Successfully validated {len(validated_data)} test cases from {len(cleaned_data)} total rows")
            
            if not validated_data:
                logger.warning("❌ No valid test cases found! Check your CSV file content.")
            
            return validated_data
            
        except FileNotFoundError as e:
            logger.error(f"❌ CSV file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load complaint workflow CSV: {e}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Data directory: {self.data_dir}")
            raise


@pytest.fixture(scope="session") 
def enhanced_data_loader() -> RobustDataLoader:
    """Enhanced data loader with robust encoding and validation capabilities."""
    return RobustDataLoader()


# ===== Test Context Fixtures =====
@pytest.fixture
def test_context(request) -> Dict[str, Any]:
    """Provide clean context for each test."""
    test_id = request.node.nodeid.split("::")[-1]
    return {
        'test_id': test_id,
        'start_time': datetime.now(),
        'test_data': {},
        'results': {},
        'response': None,
        'conversation_id': None,
        'interaction_id': None,
        'correlation_id': None,
        'step_results': []
    }


@pytest.fixture
def given_api_is_available(api_client: ChatbotAPIClient, test_context: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure API is available before test."""
    test_context['api_client'] = api_client
    _report_generator.add_step_result({
        'test_id': _current_test_id,
        'type': 'given',
        'description': 'API is available',
        'status': 'passed'
    })
    return test_context


@pytest.fixture
def given_test_data_loaded(data_loader: DataLoader, test_context: Dict[str, Any], request) -> Dict[str, Any]:
    """Load test data for scenario."""
    test_data_row = getattr(request, 'param', {}) if hasattr(request, 'param') else {}
    
    if not test_data_row and hasattr(request.node, 'callspec'):
        test_data_row = getattr(request.node.callspec, 'params', {}).get('test_data_row', {})
    
    test_context['test_data'] = test_data_row
    test_context['data_loader'] = data_loader
    
    test_case_id = test_data_row.get("test_case_id", "unknown")
    _report_generator.add_step_result({
        'test_id': _current_test_id,
        'type': 'given',
        'description': f'Test data loaded for case: {test_case_id}',
        'status': 'passed'
    })
    return test_context


# ===== CSV Data Fixtures =====
@pytest.fixture
def validate_csv_data(test_data_row):
    """Validate CSV data structure and log available columns."""
    logger = get_logger(__name__)
    
    if not test_data_row:
        logger.warning("Empty test data row received")
        return test_data_row
    
    # Clean up data - remove BOM and extra whitespace
    for key in list(test_data_row.keys()):
        clean_key = key.replace('\ufeff', '').strip()
        if clean_key != key:
            test_data_row[clean_key] = test_data_row.pop(key)
    
    # Log available chatText columns
    chat_text_columns = {}
    for i in range(1, 12):
        column_name = f'chatText{i}'
        value = str(test_data_row.get(column_name, '')).strip()
        if value and value.lower() not in ['', 'n/a', 'null', 'none', 'nan']:
            chat_text_columns[column_name] = value
    
    logger.info(f"Test case {test_data_row.get('test_case_id', 'unknown')} has {len(chat_text_columns)} chatText values")
    
    # Add metadata to test data
    test_data_row['_chattext_columns'] = chat_text_columns
    test_data_row['_has_valid_data'] = len(chat_text_columns) > 0
    
    return test_data_row


# ===== Test Generation =====
def pytest_generate_tests(metafunc):
    """Generate parameterized tests from CSV data for complaint workflow."""
    if "test_data_row" in metafunc.fixturenames:
        logger = get_logger(__name__)
        data_loader = RobustDataLoader()
        
        try:
            # Load complaint workflow CSV data with robust encoding handling
            csv_file = "complaint_data.csv"  # Updated to match the actual file name
            logger.info(f"🔄 Generating tests from CSV file: {csv_file}")
            
            complaint_data = data_loader.load_complaint_workflow_csv(csv_file)
            
            if complaint_data:
                logger.info(f"📊 Found {len(complaint_data)} valid test cases")
                
                # Generate test IDs
                test_ids = [
                    row.get("test_case_id", f"TC_{idx:03d}")
                    for idx, row in enumerate(complaint_data)
                ]
                
                # Log test cases being generated
                for test_id in test_ids:
                    logger.debug(f"  📋 {test_id}")
                
                metafunc.parametrize("test_data_row", complaint_data, ids=test_ids)
                logger.info(f"✅ Successfully generated {len(complaint_data)} parameterized test cases")
                
            else:
                logger.error("❌ No valid test cases found in CSV file")
                # Create a dummy test that will fail with a clear message
                dummy_data = [{
                    "test_case_id": "NO_VALID_DATA",
                    "error_message": "No valid test data found in CSV file"
                }]
                metafunc.parametrize("test_data_row", dummy_data, ids=["NO_VALID_DATA"])
                
        except FileNotFoundError as e:
            logger.error(f"❌ CSV file not found: {e}")
            # Create error test case
            error_data = [{
                "test_case_id": "CSV_FILE_NOT_FOUND", 
                "error_message": str(e)
            }]
            metafunc.parametrize("test_data_row", error_data, ids=["CSV_FILE_NOT_FOUND"])
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Create error test case with detailed info
            error_data = [{
                "test_case_id": "DATA_LOADING_ERROR",
                "error_message": f"{type(e).__name__}: {e}"
            }]
            metafunc.parametrize("test_data_row", error_data, ids=["DATA_LOADING_ERROR"])


# ===== Session Finish Hook =====
def pytest_sessionfinish(session, exitstatus):
    """Generate report after all tests complete."""
    logger = get_logger(__name__)
    
    if _test_results and session.config.getoption("--report"):
        try:
            # Calculate execution metadata
            metadata = {
                'environment': os.getenv('ENVIRONMENT', 'unknown'),
                'platform': platform.system(),
                'python_version': platform.python_version(),
                'total_duration': (datetime.now() - _test_start_time).total_seconds(),
                'exit_status': exitstatus
            }
            
            # Generate business report
            report_path = _report_generator.generate_report(_test_results, metadata)
            logger.info(f"Test report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
    
    # Print summary
    total_tests = len(_test_results)
    passed = sum(1 for r in _test_results if r['status'] == 'passed')
    failed = sum(1 for r in _test_results if r['status'] == 'failed')
    
    logger.info("=" * 60)
    logger.info(f"Test Execution Summary")
    logger.info(f"Total: {total_tests} | Passed: {passed} | Failed: {failed}")
    logger.info(f"Pass Rate: {passed / total_tests * 100:.1f}%" if total_tests > 0 else "N/A")
    logger.info("=" * 60)
