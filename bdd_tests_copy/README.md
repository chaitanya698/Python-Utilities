<img width="719" height="59" alt="image" src="https://github.com/user-attachments/assets/9dfa1613-50b0-4709-913a-997678f1b2e5" />

**Framework Architecture**

<img width="838" height="560" alt="image" src="https://github.com/user-attachments/assets/76c3117b-ae7a-41e0-9f4b-772aa3f6827f" />
<img width="830" height="673" alt="image" src="https://github.com/user-attachments/assets/ecd8f597-0612-4658-95f4-3abec0f40729" />
<img width="822" height="655" alt="image" src="https://github.com/user-attachments/assets/76ef6bac-fe52-4a4a-a06e-2f658f3ea6aa" />

**Core Technologies**

pytest-bdd: BDD test framework with Gherkin scenarios
Playwright APIRequestContext: High-performance API testing (no browser overhead)
Pydantic Settings: Configuration management
Oracle Database: Data validation and persistence testing
Certificate Authentication: Secure API communication

Key Features

✅ API-Only Testing: Optimized for speed 
✅ Multi-Environment Support: dev, qa, staging, production
✅ Data-Driven Testing: CSV-based test data with dynamic execution
✅ Advanced Error Injection: Comprehensive negative testing scenarios
✅ Smart Retries: Exponential backoff and custom wait strategies
✅ Database Validation: Oracle DB integration for end-to-end verification
✅ Rich Reporting: HTML reports with step-level details and API tracking
✅ Parallel Execution: Multi-worker test execution
✅ Certificate-based Auth: Secure API communication

📁 Project Structure
bdd_tests/
├── config/                          # Configuration management
│   ├── settings.py                  # Pydantic settings with env support
│   └── loader.py                    # Config loader with certificate processing
├── database/                        # Database integration
│   └── db_manager.py               # Oracle DB manager with connection pooling
├── features/                        # BDD feature files
│   ├── complaint_capture.feature   # Main complaint workflow tests
│   ├── dynamic_steps.feature       # Dynamic step execution
│   └── steps/                      # Step definitions
│       ├── test_complaint_e2e_steps.py
│       └── test_dynamic_steps.py
├── fixtures/                        # Test fixtures
│   ├── api_service.py              # Playwright API client
│   └── db_utils.py                 # Database utilities
├── data/                            # Test data
│   ├── complaint_data.csv          # Main test data
│   └── complaint_capture_data.csv  # Workflow-specific data
├── resources/                       # Request templates and configs
│   ├── initial_request.json        # API request templates
│   └── *.json                      # Error scenario templates
├── utils/                           # Utility modules
│   ├── data_loader.py              # Robust CSV/JSON loading
│   ├── error_injector.py           # Error scenario injection
│   ├── request_response_tracker.py # API call tracking
│   ├── report_generator.py         # HTML report generation
│   └── helpers.py                  # Common utilities
├── reports/                         # Generated reports
├── logs/                           # Execution logs
├── test_runner.py                  # Main execution entry point
├── conftest.py                     # pytest configuration
├── pytest.ini                     # pytest settings
└── requirements.txt                # Dependencies

🚀 **Quick Start**

1. Installation
bash# Clone the repository
git clone <repository-url>
cd bdd_tests_copy

# Create & Switch to the virtual env
 a. pythom -m venv venv
 b. \ven\source\activate 
  
# Install dependencies
pip install -r requirements.txt

# Install Playwright (API-only, no browsers needed)
python -m pip install playwright
2. Environment Configuration
Create environment-specific configuration files:
bash# QA Environment
cp .env.qa.example .env.qa

# Configure your .env.qa file
ENVIRONMENT=qa
API_BASE_URL=https://qa-api.example.com
DB_HOST=qa-db.example.com
DB_USER=qa_user
DB_PRD=your_password
CERT_PFX_PATH=certs/qa/certificate.pfx
CERT_PRD=cert_password

3. Run Tests
bash# Run all tests in QA environment
python test_runner.py --env qa

# Run smoke tests with detailed reporting
python test_runner.py --env qa --tags smoke --verbose

# Run specific feature with parallel execution
python test_runner.py --env qa --feature features/complaint_capture.feature --parallel 4

# Run with Playwright API debugging
python test_runner.py --env qa --playwright-debug --verbose -vv

🔄 **Test Execution Flow**

1. Framework Initialization
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Environment   │───▶│   Configuration  │───▶│   API Client    │
│   Selection     │    │   Loading        │    │   Initialize    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │   Certificate    │
                      │   Processing     │
                      └──────────────────┘
2. Test Data Flow
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Data      │───▶│   Data           │───▶│   Test          │
│   Loading       │    │   Validation     │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                      ┌──────────────────┐
                      │   Dynamic Step   │
                      │   Execution      │
                      └──────────────────┘
3. API Interaction Flow
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Initial       │───▶│   Workflow       │───▶│   Database      │
│   Request       │    │   Steps          │    │   Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Conversation  │    │   Step-by-Step   │    │   Complaint     │
│   ID Generated  │    │   API Calls      │    │   Persisted     │
└─────────────────┘    └──────────────────┘    └─────────────────┘


📊 **Test Execution Details**

**Data-Driven Testing**
Tests are generated from CSV file.

**Dynamic Step Execution**

The framework dynamically executes steps based on available data:
python# Only execute steps that have valid data in CSV for field_name in WORKFLOW_STEPS_ORDER:
    if is_valid_value(csv_data.get(field_name)):
        execute_workflow_step(test_context, field_name)
        
**Error Injection**
Comprehensive negative testing with built-in error scenarios:
python# Available error scenarios
error_scenarios = [
    'missing_conversation_id',
    'invalid_request_type', 
    'empty_payload',
    'invalid_date_format',
    'service_unavailable'
]

📈 **Reporting & Analytics**
HTML Reports
Rich HTML reports with:

**Executive summary with key metrics**
Test results with expandable step details
API request/response tracking
Performance metrics
Environment information

**Step-Level Tracking**
Every BDD step is tracked with:

Execution status (passed/failed/skipped)
Request/response details
Timing information
Error details (if any)

**API Call Tracking**
All API interactions are logged:

Request headers and payload
Response status and data
Correlation IDs for tracing
Performance metrics

🛠️ **Configuration Options**
Command Line Options
bashpython test_runner.py [OPTIONS]

**Options**:
  --env {dev,qa,staging,production}  Environment to test against
  --tags TEXT                        Pytest markers (e.g., 'smoke', 'regression')
  --feature TEXT                     Specific feature file to run
  --parallel N                       Run tests in parallel with N workers
  --api-timeout SECONDS             Override API timeout
  --api-retry-count COUNT           Override retry count
  --playwright-debug                 Enable Playwright API debugging
  --verbose, -v                     Increase verbosity (-vv for more)
  --max-failures N                   Stop after N failures
  --dry-run                         Show what would run without executing

**Environment Variables**
ENVIRONMENT=qa
API_BASE_URL=https://api.example.com
API_TIMEOUT=45
API_RETRY_COUNT=3

# Database Configuration  
DB_HOST=DB_HOST
DB_PORT=DB_PORT
DB_USER=DB_USER
DB_PRD=DB_PRD
DB_SERVICE_NAME=DB_SERVICE_NAME

# Certificate Configuration
CERT_PFX_PATH=certs/certificate.pfx
CERT_PRD=cert_password

# Logging Configuration
LOG_LEVEL=INFO
ENABLE_DETAILED_LOGGING=true
VERIFY_SSL=true

🔧 **Advanced Features**
Smart Retries
Exponential backoff strategy
Configurable retry counts
Custom wait strategies for different scenarios

**Certificate Management**

Automatic PFX to PEM conversion
Secure credential handling
Multi-environment certificate support

**Database Integration**

Oracle connection pooling
Transaction management
Query result validation
Cleanup utilities

**Parallel Execution**

Multi-worker test execution
Load balancing with worksteal distribution
Resource isolation between workers

📋 **Best Practices**

**Test Data Management**

Use meaningful test case IDs
Validate CSV data before execution
Separate positive and negative test data

**Error Handling**

Implement comprehensive error scenarios
Use correlation IDs for request tracing
Log detailed error information

**Trouble Shooting & Debugging**

Check the logs in the logs/ directory
Review HTML reports for detailed execution information
Use --playwright-debug for API troubleshooting
Enable verbose logging with -vv for detailed output

